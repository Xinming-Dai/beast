"""Command modules for the beast CLI."""

from beast.cli.commands import extract, extract_3d, extract_sable, predict, train

# dictionary of all available commands
COMMANDS = {
    'extract': extract,              # 2D frame extraction
    'train': train,                  # model training
    'predict': predict,              # model inference on images and videos
    'extract_3d': extract_3d,        # 3D frame extraction and segmentation
    'extract_sable': extract_sable,  # SABLE IBL stereo extraction pipeline
}
