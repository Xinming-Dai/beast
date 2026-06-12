#!/usr/bin/env python3
"""
Stage 1/2 smoke + evaluation launcher for Cheese3D LP3D Kabsch.

Modes:
  --smoke    : 4-step training smoke (NOT evaluation)
  --eval     : zero-shot NVS evaluation (metrics + visuals, no weight updates)
  --finetune : fine-tune pilot with NVS protocol (full training loop + val metrics)

Smoke modes (only valid with --smoke):
  clean : verify gs_reg_loss is non-zero and finite
  debug : verify debug_pcd/batch_000/ PLY artifacts

Eval mode:
  - Loads model (optionally from --resume_ckpt)
  - Runs forward pass over val/train splits
  - Computes l2 / psnr / gs_reg / perceptual metrics
  - Saves render-vs-target PNG grids

Finetune mode:
  - Loads erayzer_dl3dv.pt checkpoint (if --resume_ckpt)
  - Fine-tunes on Cheese3D NVS protocol
  - Logs train/val metrics at intervals
  - Saves convergence curves
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

try:
    import tqdm
except ImportError:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

HF_DINO_DIR = (
    "/data/jqh/pretrained_checkpoints/E-RayZer-private/checkpoints/"
    "dinov3-vitb16-pretrain-lvd1689m"
)
os.environ["HF_HOME"] = HF_DINO_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
os.environ["PATH"] = "/home/jqh/miniconda3/envs/neuro/bin:" + os.environ.get("PATH", "")
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba-cache"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"

from beast.inference import save_gaussian_pointclouds  # noqa: E402
from beast.io import load_config  # noqa: E402
from beast.models.model_utils.data_utils import collate_with_correspondence_padding  # noqa: E402
from beast.models.model_utils.train_vis import save_training_visuals  # noqa: E402
from beast.models.sable import Sable  # noqa: E402
from beast.train_sable import train_sable  # noqa: E402


def _progress_bars_enabled() -> bool:
    """Return whether interactive tqdm bars should be shown.

    In nohup/CI logs, tqdm carriage returns make train/val bars overlap and
    bloat log files. Keep bars only for interactive terminals unless explicitly
    overridden.
    """
    override = os.environ.get("SABLE_PROGRESS_BARS", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    return sys.stderr.isatty()


def _tqdm_iter(iterable, **kwargs):
    if tqdm is None:
        return iterable
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("disable", not _progress_bars_enabled())
    return tqdm.tqdm(iterable, **kwargs)


def _make_tqdm_bar(**kwargs):
    if tqdm is None or not _progress_bars_enabled():
        return None
    kwargs.setdefault("file", sys.stderr)
    return tqdm.tqdm(**kwargs)


def _extract_model_state_dict(raw_checkpoint: dict) -> dict:
    """Extract model weights from supported checkpoint layouts."""
    for key in ("state_dict", "model_state_dict", "model"):
        state_dict = raw_checkpoint.get(key)
        if isinstance(state_dict, dict):
            return state_dict
    return raw_checkpoint


def _save_effective_run_config(output_dir: Path, args: argparse.Namespace, config: dict) -> None:
    """Persist the exact CLI args and config used for this run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(output_dir / "effective_config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cheese3D LP3D Stage 1/2 smoke or evaluation")
    parser.add_argument("--config", default="beast/configs/sable_cheese3d_lp3d.yaml",
                        help="Path to config file relative to workspace root")
    parser.add_argument("--session", default=None,
                        help="Session to use; None = all sessions")
    parser.add_argument("--correspondence_cache", default=None,
                        help="Path to LP3D correspondence cache")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true",
                      help="Run a 4-step training smoke (NOT evaluation)")
    group.add_argument("--eval", action="store_true",
                      help="Run zero-shot NVS evaluation (metrics + visuals, no weight updates)")
    group.add_argument("--finetune", action="store_true",
                      help="Fine-tune pilot with NVS protocol (full training + val metrics)")
    parser.add_argument(
        "--smoke_mode", choices=("clean", "debug"), default="debug",
        help="'clean': verify gs_reg_loss non-zero finite; 'debug': verify debug_pcd artifacts")
    parser.add_argument("--smoke_output_dir",
                        default="./outputs/cheese3d_stage1_single_session/smoke",
                        help="Output dir for smoke artifacts")
    parser.add_argument("--eval_splits", nargs="+", default=["val"],
                        help="Splits to evaluate (e.g. val train)")
    parser.add_argument("--eval_output_dir",
                        default="./outputs/cheese3d_eval",
                        help="Output dir for eval metrics and visuals")
    parser.add_argument("--finetune_output_dir",
                        default="./outputs/cheese3d_finetune",
                        help="Output dir for fine-tune runs")
    parser.add_argument("--vis_samples", type=int, default=2,
                        help="Number of samples to visualize per split")
    parser.add_argument("--vis_views", type=int, default=None,
                        help="Max target views in visuals (default: num_target_views)")
    parser.add_argument("--save_pointclouds", action="store_true",
                        help="In --eval mode, save Gaussian point clouds as PLY files")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Stop after N batches per split (for quick eval)")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Pretrained checkpoint (e.g. erayzer_dl3dv.pt)")
    parser.add_argument("--reset_training_state", action="store_true", default=False,
                        help="Reset optimizer/scheduler state when loading checkpoint")
    parser.add_argument("--init_gs", type=str, default=None, choices=["true", "false"],
                        help="Override init_gs in gaussians config")
    parser.add_argument("--correspondence_mode", type=str, default=None, choices=["cache", "none"],
                        help="Override correspondence_mode (cache=LP3D, none=no correspondence)")
    parser.add_argument("--gs_reg_loss_weight", type=float, default=None,
                        help="Override training.gs_reg_loss_weight")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Max training steps for --finetune mode (default: 200)")
    parser.add_argument("--val_every", type=int, default=50,
                        help="Validate every N steps (default: 50)")
    parser.add_argument("--sample_limit", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"torch: {torch.__version__} cuda: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cache_root_arg = args.correspondence_cache
    if cache_root_arg is not None:
        cache_root = Path(cache_root_arg)
        if not cache_root.exists():
            raise FileNotFoundError(f"Correspondence cache not found: {cache_root}")
    else:
        cache_root = None

    config_path = (REPO_ROOT / args.config.lstrip('beast/').lstrip('/')).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(str(config_path))

    if args.session:
        config["training"]["sessions"] = [args.session]

    if args.resume_ckpt:
        config["training"]["resume_ckpt"] = str(Path(args.resume_ckpt).resolve())
        config["training"]["reset_training_state"] = args.reset_training_state
        print(f"resume_ckpt: {config['training']['resume_ckpt']}")
        print(f"reset_training_state: {config['training']['reset_training_state']}")

    if args.gs_reg_loss_weight is not None:
        config["training"]["gs_reg_loss_weight"] = float(args.gs_reg_loss_weight)
        print(f"gs_reg_loss_weight override: {config['training']['gs_reg_loss_weight']}")

    if cache_root is not None:
        config["model"]["merge_pcd"]["correspondence_cache_root"] = str(cache_root)
    config["model"]["vda"]["mode"] = "online"

    if args.smoke:
        _run_smoke(args, config, cache_root, config_path)
    elif args.eval:
        _run_eval(args, config, cache_root)
    elif args.finetune:
        _run_finetune(args, config, cache_root, config_path)


def _run_smoke(args, config, cache_root, config_path) -> None:
    import subprocess

    validator_script = (
        REPO_ROOT / "scripts" / "sable_scripts" / "validate_cheese3d_stage1_cache.py"
    )
    python_exe = Path(os.environ["PATH"].split(":")[0]) / "python"
    validate_cmd = [
        str(python_exe), str(validator_script),
        "--config", str(config_path),
        "--cache_root", str(cache_root),
        "--sessions", args.session or "",
        "--sample_limit", str(args.sample_limit),
    ]
    subprocess.run(validate_cmd, check=True)

    config["training"]["num_workers"] = 0
    config["training"]["val_every"] = 0
    config["training"]["save_visuals"] = False
    config["training"]["max_frames_per_session"] = 4

    smoke_output_dir = (REPO_ROOT / args.smoke_output_dir).resolve()
    config["training"]["max_fwdbwd_passes"] = 4
    config["training"]["checkpoint_dir"] = str(smoke_output_dir)
    config["model"]["merge_pcd"]["debug_merged_pcd"] = args.smoke_mode == "debug"
    _save_effective_run_config(smoke_output_dir, args, config)

    output_dir = Path(config["training"]["checkpoint_dir"])
    print(f"Output: {output_dir}")
    print(f"init_gs: {config['model']['gaussians']['init_gs']}")
    print(f"correspondence_mode: {config['training'].get('correspondence_mode')}")
    print(f"gs_reg_loss_weight: {config['training'].get('gs_reg_loss_weight')}")
    print(f"smoke_mode: {args.smoke_mode}")
    print(f"resume_ckpt: {config['training'].get('resume_ckpt', 'None')}")

    print("Initializing Sable model...")
    model = Sable(config)
    print("Model initialized successfully.")

    train_sable(config, model, output_dir=str(output_dir))

    if args.smoke_mode == "clean":
        print("Clean smoke finished. Verify non-zero, finite gs_reg_loss.")
    else:
        print("Debug smoke finished. Verify debug_pcd/batch_000/ PLY and overlay artifacts.")


def _run_eval(args, config, cache_root) -> None:
    """Zero-shot NVS evaluation: load checkpoint (optional) + trainer.validate()."""
    eval_output_dir = (REPO_ROOT / args.eval_output_dir).resolve()
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = eval_output_dir / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)

    config["training"]["num_workers"] = 0
    config["training"]["save_visuals"] = True
    config["training"]["checkpoint_dir"] = str(eval_output_dir)
    config["inference"] = True  # disables view randomization, forces rendering
    if cache_root is not None:
        config["model"]["merge_pcd"]["correspondence_cache_root"] = str(cache_root)
    if args.correspondence_mode is not None:
        config["training"]["correspondence_mode"] = args.correspondence_mode
        print(f"correspondence_mode override: {args.correspondence_mode}")
    if args.init_gs is not None:
        config["model"]["gaussians"]["init_gs"] = args.init_gs == "true"
        print(f"init_gs override: {config['model']['gaussians']['init_gs']}")
    _save_effective_run_config(eval_output_dir, args, config)

    vis_views = args.vis_views or config["training"].get("num_target_views", 2)

    print(f"\nOutput dir: {eval_output_dir}")
    print(f"Eval splits: {args.eval_splits}")
    print(f"vis_samples: {args.vis_samples}, vis_views: {vis_views}")
    print(f"resume_ckpt: {config['training'].get('resume_ckpt', 'None')}")
    print(f"init_gs: {config['model']['gaussians']['init_gs']}")
    print(f"correspondence_mode: {config['training'].get('correspondence_mode')}")
    print(f"gs_reg_loss_weight: {config['training'].get('gs_reg_loss_weight')}")
    regime = config['training'].get('ibl_training_regime', 'two_input_reconstruction')
    print(f"ibl_training_regime: {regime}")
    print(f"num_input_views: {config['training'].get('num_input_views')}, "
          f"num_target_views: {config['training'].get('num_target_views')}")

    resume_ckpt = config["training"].get("resume_ckpt")

    print("Initializing Sable model (inference=True)...")
    model = Sable(config)
    print("Model initialized.")

    if resume_ckpt:
        print(f"Loading checkpoint: {resume_ckpt}")
        raw = torch.load(resume_ckpt, map_location='cpu', weights_only=False)
        state_dict = _extract_model_state_dict(raw)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print("  Checkpoint loaded.")

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dataset_name = str(config['training'].get(
        'dataset_name', 'beast.data.ibl_dataset.IBLDataset'))
    module_path, class_name = dataset_name.rsplit('.', 1)
    module = importlib.import_module(module_path)
    DatasetClass = getattr(module, class_name)

    loaders = {}
    for split in args.eval_splits:
        include = ['train'] if split == 'train' else ['val']
        dataset = DatasetClass(config, include_splits=include)
        batch_size = int(config['training'].get('batch_size_per_gpu', 1))

        if args.max_batches:
            max_records = args.max_batches * batch_size

            class _TruncatedDataset:
                def __init__(self, ds, limit):
                    self._ds = ds
                    self._limit = limit

                def __len__(self):
                    return min(len(self._ds), self._limit)

                def __getitem__(self, idx):
                    return self._ds[idx]

            dataset = _TruncatedDataset(dataset, max_records)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_with_correspondence_padding,
            drop_last=False,
        )
        loaders[split] = loader
        print(f"  [{split}] {len(loader)} batches, {len(dataset)} records")

    all_results = {}
    for split, loader in loaders.items():
        print(f"\n{'=' * 60}")
        print(f"Eval: split={split}  ({len(loader)} batches)")
        print(f"{'=' * 60}")

        l2_vals, psnr_vals, gs_vals, perc_vals = [], [], [], []
        saved_vis_count = 0
        saved_ply_count = 0

        def move_to(batch, device):
            if isinstance(batch, torch.Tensor):
                return batch.to(device)
            elif isinstance(batch, dict):
                return {k: move_to(v, device) for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                return type(batch)(move_to(x, device) for x in batch)
            return batch

        for batch_idx, batch in enumerate(
            _tqdm_iter(loader, desc=f"eval[{split}]", leave=False, mininterval=5)
        ):
            batch = move_to(batch, model.device)

            with torch.no_grad():
                data_dict = model.get_model_outputs(batch_dict=batch)
                loss_out, log_list = model.compute_loss(stage='val', **data_dict)

            for item in log_list:
                name, val = item['name'], item['value']
                if 'val_l2' == name:
                    l2_vals.append(val.detach().cpu().item())
                elif 'val_psnr' == name:
                    psnr_vals.append(val.detach().cpu().item())
                elif 'val_gs_reg' == name:
                    gs_vals.append(val.detach().cpu().item())
                elif 'val_perceptual' == name:
                    perc_vals.append(val.detach().cpu().item())

            if saved_vis_count < args.vis_samples:
                try:
                    result_ns = SimpleNamespace(**data_dict)
                    vis_out_dir = vis_dir / split
                    vis_out_dir.mkdir(parents=True, exist_ok=True)
                    paths = save_training_visuals(
                        vis_out_dir,
                        result=result_ns,
                        batch=batch,
                        step=0,
                        max_samples=args.vis_samples - saved_vis_count,
                        max_views=vis_views,
                    )
                    if paths:
                        saved_vis_count += len(paths)
                        print(f"  [{split}] Saved visuals: {paths[0]}")
                except Exception as exc:
                    print(f"  [{split}] Visual save failed: {exc}")

            if args.save_pointclouds and saved_ply_count < args.vis_samples:
                try:
                    ply_paths = save_gaussian_pointclouds(
                        data_dict,
                        eval_output_dir / split,
                        batch_idx,
                        max_samples=args.vis_samples - saved_ply_count,
                    )
                    if ply_paths:
                        saved_ply_count += len(ply_paths)
                        print(f"  [{split}] Saved point cloud: {ply_paths[0]}")
                except Exception as exc:
                    print(f"  [{split}] Point cloud save failed: {exc}")

        n = max(len(l2_vals), 1)
        agg = {
            'val_l2': sum(l2_vals) / n,
            'val_psnr': sum(psnr_vals) / n,
            'val_gs_reg': sum(gs_vals) / n,
            'val_perceptual': sum(perc_vals) / n if perc_vals else 0.0,
        }
        print(f"\n  [{split}] Aggregated over {n} batches:")
        for k, v in sorted(agg.items()):
            print(f"    {k}: {v:.6f}")
        all_results[split] = agg

    primary_split = "val" if "val" in all_results else next(iter(all_results), None)
    if primary_split is not None:
        metrics_path = eval_output_dir / "metrics.json"
        metrics_payload = {
            **all_results[primary_split],
            "primary_split": primary_split,
            "eval_splits": list(all_results.keys()),
            "max_batches": args.max_batches,
            "resume_ckpt": config["training"].get("resume_ckpt"),
            "correspondence_cache": str(cache_root) if cache_root is not None else None,
            "init_gs": bool(config["model"]["gaussians"]["init_gs"]),
            "correspondence_mode": config["training"].get("correspondence_mode"),
            "gs_reg_loss_weight": float(config["training"].get("gs_reg_loss_weight", 0.0)),
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_payload, f, indent=2)

    metrics_by_split_path = eval_output_dir / "metrics_by_split.json"
    with open(metrics_by_split_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"FINAL EVALUATION RESULTS  (output: {eval_output_dir})")
    print(f"{'=' * 60}")
    print(f"{'Split':<10} {'l2':>12} {'psnr':>10} {'gs_reg':>12} {'perceptual':>14}")
    print("-" * 62)
    for split, m in all_results.items():
        l2 = m.get('val_l2', m.get('test_l2', m.get('train_l2', float('nan'))))
        psnr = m.get('val_psnr', m.get('test_psnr', m.get('train_psnr', float('nan'))))
        gs = m.get('val_gs_reg', m.get('test_gs_reg', m.get('train_gs_reg', float('nan'))))
        perc = m.get(
            'val_perceptual',
            m.get('test_perceptual', m.get('train_perceptual', float('nan'))),
        )
        print(f"{split:<10} {l2:>12.6f} {psnr:>10.4f} {gs:>12.6f} {perc:>14.6f}")
    print(f"\nMetrics: {eval_output_dir / 'metrics.json'}")
    print(f"\nVisuals: {vis_dir}")


def _run_finetune(args, config, cache_root, config_path) -> None:
    """Fine-tune pilot: load checkpoint + NVS training loop + periodic val metrics."""
    import subprocess
    from collections import defaultdict

    finetune_output_dir = (REPO_ROOT / args.finetune_output_dir).resolve()
    finetune_output_dir.mkdir(parents=True, exist_ok=True)

    # Apply overrides
    config["training"]["num_workers"] = 0
    config["training"]["max_fwdbwd_passes"] = args.max_steps
    config["training"]["val_every"] = args.val_every
    config["training"]["checkpoint_dir"] = str(finetune_output_dir)
    config["training"]["save_visuals"] = True
    if cache_root is not None:
        config["model"]["merge_pcd"]["correspondence_cache_root"] = str(cache_root)
    config["model"]["vda"]["mode"] = "online"

    if args.correspondence_mode is not None:
        config["training"]["correspondence_mode"] = args.correspondence_mode
        print(f"correspondence_mode override: {args.correspondence_mode}")
    if args.init_gs is not None:
        config["model"]["gaussians"]["init_gs"] = args.init_gs == "true"
        print(f"init_gs override: {config['model']['gaussians']['init_gs']}")
    _save_effective_run_config(finetune_output_dir, args, config)

    # Validate cache if using correspondence
    if config["training"].get("correspondence_mode") == "cache":
        validator_script = (
            REPO_ROOT / "scripts" / "sable_scripts" / "validate_cheese3d_stage1_cache.py"
        )
        python_exe = Path(os.environ["PATH"].split(":")[0]) / "python"
        validate_cmd = [
            str(python_exe), str(validator_script),
            "--config", str(config_path),
            "--cache_root", str(cache_root),
            "--sessions", args.session or "",
            "--sample_limit", str(args.sample_limit),
        ]
        subprocess.run(validate_cmd, check=True)

    # Checkpoint save + fixed-eval helpers (need model/optimizer/scaler in closure)
    def save_checkpoint(step_num):
        ckpt_path = finetune_output_dir / f"checkpoint_step_{step_num:05d}.pt"
        torch.save({
            'step': step_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, ckpt_path)
        return ckpt_path

    def fixed_eval_checkpoint(step_num, ckpt_path):
        """Load ckpt, run full fixed eval over val split, restore training state."""
        eval_output_dir = finetune_output_dir / "fixed_eval" / f"step_{step_num:05d}"
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        saved_model = model.state_dict()
        saved_opt = optimizer.state_dict()
        saved_scaler = scaler.state_dict()

        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(raw['model_state_dict'])
        model.eval()

        l2_vals, psnr_vals, gs_vals, perc_vals = [], [], [], []
        with torch.no_grad():
            for vbatch in _tqdm_iter(val_loader, desc="fixed_eval", leave=False, mininterval=5):
                vbatch = move_to(vbatch, device)
                data_dict = model.get_model_outputs(batch_dict=vbatch)
                _, log_list = model.compute_loss(stage='val', **data_dict)
                for item in log_list:
                    name, val = item['name'], item['value']
                    if 'val_l2' == name:
                        l2_vals.append(val.detach().cpu().item())
                    elif 'val_psnr' == name:
                        psnr_vals.append(val.detach().cpu().item())
                    elif 'val_gs_reg' == name:
                        gs_vals.append(val.detach().cpu().item())
                    elif 'val_perceptual' == name:
                        perc_vals.append(val.detach().cpu().item())

        n = max(len(l2_vals), 1)
        metrics = {
            'step': step_num,
            'val_l2': sum(l2_vals) / n,
            'val_psnr': sum(psnr_vals) / n,
            'val_gs_reg': sum(gs_vals) / n,
            'val_perceptual': sum(perc_vals) / n if perc_vals else 0.0,
        }

        model.load_state_dict(saved_model)
        optimizer.load_state_dict(saved_opt)
        scaler.load_state_dict(saved_scaler)
        model.train()

        eval_path = eval_output_dir / "metrics.json"
        with open(eval_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        return metrics

    # Checkpoint eval plan: at which training steps to save ckpt + fixed eval
    eval_at_steps = set(range(0, args.max_steps + 1, 250))
    eval_at_steps.discard(0)
    eval_at_steps.add(args.max_steps)

    print(f"\nOutput dir: {finetune_output_dir}")
    print(f"max_steps: {args.max_steps}, val_every: {args.val_every}")
    print(f"Checkpoint fixed-eval steps: {sorted(eval_at_steps)}")
    print(f"resume_ckpt: {config['training'].get('resume_ckpt', 'None')}")
    print(f"init_gs: {config['model']['gaussians']['init_gs']}")
    print(f"correspondence_mode: {config['training'].get('correspondence_mode')}")
    print(f"gs_reg_loss_weight: {config['training'].get('gs_reg_loss_weight')}")
    regime = config['training'].get('ibl_training_regime', 'two_input_reconstruction')
    print(f"ibl_training_regime: {regime}")
    print(f"num_input_views: {config['training'].get('num_input_views')}, "
          f"num_target_views: {config['training'].get('num_target_views')}")

    print("Initializing Sable model...")
    model = Sable(config)
    print("Model initialized.")

    resume_ckpt = config["training"].get("resume_ckpt")
    if resume_ckpt:
        print(f"Loading pretrained weights for fine-tune: {resume_ckpt}")
        raw = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        state_dict = _extract_model_state_dict(raw)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Matched keys: {len(state_dict) - len(unexpected)}/{len(state_dict)}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:8]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:8]}...")
        print("  Pretrained weights loaded. Optimizer/scaler start fresh in this fine-tune loop.")

    # Build train and val dataloaders
    dataset_name = str(config['training'].get(
        'dataset_name', 'beast.data.cheese3d_dataset.Cheese3DDataset'))
    module_path, class_name = dataset_name.rsplit('.', 1)
    module = importlib.import_module(module_path)
    DatasetClass = getattr(module, class_name)

    train_dataset = DatasetClass(config, include_splits=['train'])
    val_dataset = DatasetClass(config, include_splits=['val'])

    batch_size = int(config['training'].get('batch_size_per_gpu', 1))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_with_correspondence_padding, drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_with_correspondence_padding, drop_last=False,
    )
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    # Optimizer + scaler
    from torch.amp import GradScaler, autocast
    cfg_opt = config['optimizer']
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg_opt['lr'],
        betas=(cfg_opt['beta1'], cfg_opt['beta2']),
        weight_decay=cfg_opt['wd'],
    )
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # Tracking
    history = defaultdict(list)
    step = 0

    def move_to(batch, dev):
        if isinstance(batch, torch.Tensor):
            return batch.to(dev)
        elif isinstance(batch, dict):
            return {k: move_to(v, dev) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(move_to(x, dev) for x in batch)
        return batch

    def validate():
        model.eval()
        l2_vals, psnr_vals, gs_vals, perc_vals = [], [], [], []
        with torch.no_grad():
            for vbatch in _tqdm_iter(val_loader, desc="val", leave=False, mininterval=5):
                vbatch = move_to(vbatch, device)
                data_dict = model.get_model_outputs(batch_dict=vbatch)
                _, log_list = model.compute_loss(stage='val', **data_dict)
                for item in log_list:
                    name, val = item['name'], item['value']
                    if 'val_l2' == name:
                        l2_vals.append(val.detach().cpu().item())
                    elif 'val_psnr' == name:
                        psnr_vals.append(val.detach().cpu().item())
                    elif 'val_gs_reg' == name:
                        gs_vals.append(val.detach().cpu().item())
                    elif 'val_perceptual' == name:
                        perc_vals.append(val.detach().cpu().item())
        n = max(len(l2_vals), 1)
        return {
            'val_l2': sum(l2_vals) / n,
            'val_psnr': sum(psnr_vals) / n,
            'val_gs_reg': sum(gs_vals) / n,
            'val_perceptual': sum(perc_vals) / n if perc_vals else 0.0,
        }

    print(f"\nStarting fine-tune: {args.max_steps} steps, val every {args.val_every}")

    initial_val = validate()
    model.train()
    print(f"\n  step=0/{args.max_steps}  "
          f"train_l2=nan  "
          f"val_l2={initial_val['val_l2']:.6f}  "
          f"val_psnr={initial_val['val_psnr']:.4f}  "
          f"val_gs_reg={initial_val['val_gs_reg']:.6f}  "
          f"val_perceptual={initial_val['val_perceptual']:.6f}")
    for k, v in initial_val.items():
        history[k].append(v)

    # Training loop
    import itertools
    data_iter = itertools.cycle(train_loader)

    val_interval = args.val_every
    _epoch_start = time.time()

    pbar = _make_tqdm_bar(
        total=args.max_steps,
        initial=0,
        desc="train",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
    )

    _last_metrics = {}

    while step < args.max_steps:
        batch = next(data_iter)
        batch = move_to(batch, device)

        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            data_dict = model.get_model_outputs(batch_dict=batch)
            loss_out, log_list = model.compute_loss(stage='train', **data_dict)
        scaler.scale(loss_out).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Log train metrics
        train_l2 = None
        for item in log_list:
            name, val = item['name'], item['value']
            if 'train_l2' == name:
                train_l2 = val.detach().cpu().item()
            history[name].append(val.detach().cpu().item())

        next_step = step + 1
        should_fixed_eval = next_step in eval_at_steps

        if next_step % val_interval == 0 and not should_fixed_eval:
            val_metrics = validate()
            model.train()
            _last_metrics = {
                'train_l2': train_l2,
                'val_l2': val_metrics['val_l2'],
                'val_psnr': val_metrics['val_psnr'],
                'val_gs_reg': val_metrics['val_gs_reg'],
                'val_perceptual': val_metrics['val_perceptual'],
            }
            print(f"\n  step={next_step}/{args.max_steps}  "
                  f"train_l2={train_l2:.6f}  "
                  f"val_l2={val_metrics['val_l2']:.6f}  "
                  f"val_psnr={val_metrics['val_psnr']:.4f}  "
                  f"val_gs_reg={val_metrics['val_gs_reg']:.6f}  "
                  f"val_perceptual={val_metrics['val_perceptual']:.6f}")
            for k, v in val_metrics.items():
                history[k].append(v)

        # Checkpoint + fixed eval at planned steps
        if should_fixed_eval:
            ckpt_path = save_checkpoint(next_step)
            print(f"\n  === Fixed eval @ step {next_step} ===")
            fe_metrics = fixed_eval_checkpoint(next_step, ckpt_path)
            _last_metrics = {
                'train_l2': train_l2,
                'val_l2': fe_metrics['val_l2'],
                'val_psnr': fe_metrics['val_psnr'],
                'val_gs_reg': fe_metrics['val_gs_reg'],
                'val_perceptual': fe_metrics['val_perceptual'],
            }
            print(f"  [fixed] val_l2={fe_metrics['val_l2']:.6f}  "
                  f"val_psnr={fe_metrics['val_psnr']:.4f}  "
                  f"val_gs_reg={fe_metrics['val_gs_reg']:.6f}  "
                  f"val_perceptual={fe_metrics['val_perceptual']:.6f}")
            for k, v in fe_metrics.items():
                if k != 'step':
                    history[k].append(v)

        step += 1

        if pbar is not None:
            elapsed = time.time() - _epoch_start
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            suffix = ""
            if _last_metrics:
                suffix = (
                    f" psnr={_last_metrics.get('val_psnr', 0):.2f}"
                    f" l2={_last_metrics.get('val_l2', 0):.4f}"
                    f" ({sps:.1f} step/s)"
                )
            pbar.set_postfix_str(suffix, refresh=False)
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Save convergence history
    hist_path = finetune_output_dir / "convergence_history.json"
    with open(hist_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

    print(f"\nFine-tune complete. History saved to: {hist_path}")

    # Final validation
    val_metrics = validate()
    print(f"\n{'='*60}")
    print(f"FINAL VAL METRICS  (output: {finetune_output_dir})")
    print(f"{'='*60}")
    for k, v in sorted(val_metrics.items()):
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
