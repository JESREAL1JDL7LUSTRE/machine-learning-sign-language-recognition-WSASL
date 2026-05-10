[← Back to Main README](../README.md)

# dataset/

This directory contains the raw sign language videos organized by class folder.

## Structure
```text
dataset/
├── hello/
│   ├── video1.mp4
│   └── video2.mp4
├── thank_you/
│   └── ...
└── ...
```

## Usage
The videos in this directory are read by the `preprocessing/extract.py` script to generate skeleton keypoints.
Do NOT place processed `.npy` files or code in this directory.