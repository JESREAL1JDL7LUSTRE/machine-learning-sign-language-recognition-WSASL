# Preprocessing Module

## Purpose
Transforms raw dataset inputs into usable features for the model.

## Responsibilities
- Video → frame extraction
- Frame sampling (e.g., 8 or 16 frames per video)
- Pose/keypoint extraction (e.g., MediaPipe)
- Data normalization (scaling coordinates, etc.)

## Input
- Raw videos or frames from dataset/

## Output
- Processed features (e.g., keypoints, tensors)
- Saved into /data/processed

## Notes
- This step is usually done ONCE and saved
- Avoid recomputing preprocessing during training
- Keep transformations consistent between train and test

## Future Extensions
- Data augmentation (rotation, noise, etc.)
- Multi-modal features (pose + RGB)