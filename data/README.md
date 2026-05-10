[← Back to Main README](../README.md)

# data/

This directory is used as a local cache for pre-trained model weights downloaded by the extraction backends.

## Structure
- `*.task` → MediaPipe model weights (e.g., `pose_landmarker_heavy.task`, `hand_landmarker.task`).
- `*.pt` → YOLO model weights (e.g., `yolov8n-pose.pt`).

## Notes
- These files are automatically downloaded by `preprocessing/extract.py` when running for the first time.
- They are generally ignored by Git to save repository space.