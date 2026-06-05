# Cheese3D Stage 1 data-layout inspection

Inspected before implementing any Stage 1 data-processing changes.

## Dataset root
- Root: `/home/jqh/NeuralWorkshops/E-RayZer-private/data/cheese3d_cam`
- Frame dataset dir used by `Cheese3DDataset`: `/home/jqh/NeuralWorkshops/E-RayZer-private/data/cheese3d_cam/cheese3d_cam`
- Metadata file: `/home/jqh/NeuralWorkshops/E-RayZer-private/data/cheese3d_cam/cheese3d_cam/info.json`

## Confirmed layout facts
- `info.json` lists 11 sessions and available views `BC, L, R, TC, TL, TR`.
- Each session directory contains exactly six per-view directories plus `selected_frames.csv`.
- Example session: `20231031_B20_chew_bl_000/`
  - `BC/`, `L/`, `R/`, `TC/`, `TL/`, `TR/`
  - `selected_frames.csv`
- Inside each per-view directory, files are interleaved as:
  - `img00000000.png`
  - `img00000000.npy`
  - `img00000001.png`
  - `img00000001.npy`
  - ...
- For `20231031_B20_chew_bl_000`, each of the six views has 512 PNGs and 512 NPY camera files.
- For `20231031_B20_chew_temperature_000`, each of the six views has 632 PNGs and matching frame ids.
- On sampled sessions, the frame-id intersection across all six views is exact:
  - `20231031_B20_chew_bl_000`: common indices `0..511` (512 frames)
  - `20231031_B20_chew_temperature_000`: common indices `0..631` (632 frames)

## selected_frames.csv facts
- Real file format is one filename per row, no header.
- Example rows:
  - `img00000000.png`
  - `img00000001.png`
  - `img00000002.png`
- Therefore the current dataset parser assumption `skip one header, parse int(parts[0])` is incorrect for this dataset.

## Masks and videos
- Segmentation masks live under:
  `/home/jqh/NeuralWorkshops/E-RayZer-private/data/cheese3d_cam/segmentation_masks`
- There are 66 mask directories, one per session-view pair, named like:
  - `20231031_B20_chew_bl_000_BC_08-38-25`
- Each mask directory contains `masks/mask00000000.png`, `mask00000001.png`, ...
- Videos live under:
  `/home/jqh/NeuralWorkshops/E-RayZer-private/data/cheese3d_cam/videos`
- There are 66 MP4 files, one per session-view pair, named like:
  - `20231031_B20_chew_bl_000_BC_08-38-25.mp4`

## Stage 1 implications
- LP3D inference should use PNGs from all six views in checkpoint order `BC, L, R, TC, TL, TR`.
- Frame alignment should be based on PNG frame ids and the intersection across all required views.
- `selected_frames.csv` parsing must support filename rows with no header.
- Cache bundles must continue to be keyed by integer PNG frame id: `pair_{frame_idx:06d}`.
- Dataset-side coordinate rescaling should remain the only `320x256 -> 320x320` scaling step.
