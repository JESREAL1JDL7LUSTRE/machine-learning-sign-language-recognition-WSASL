# System Analysis: Preprocessing

## Overview
The `preprocessing` pipeline translates raw MP4 sign language videos into a standardized graph-based mathematical format required for the ST-GCN models.

## Pipeline Steps
1. **Extraction (`extract.py`)**: Uses Google's MediaPipe Pose and Hand models. For each frame, it detects 33 pose landmarks and 21 hand landmarks per hand, outputting `(X, Y)` spatial coordinates. Unseen joints are zero-padded.
2. **Missing Data Handling (`normalize.py`)**: Temporally interpolates coordinates to fill in missing frames where MediaPipe failed to detect hands/body.
3. **Filtering (`normalize.py`)**: Discards irrelevant joints (e.g., lower body, face mesh). The system filters the 150 features down to 102 relevant features (51 joints).
4. **Spatial Normalization (`normalize.py`)**: Ensures spatial invariance so the model doesn't care where the signer stands or their body proportions.
   - **Centering**: All coordinates are shifted relative to the mid-shoulder point.
   - **Scaling**: All distances are divided by the shoulder width.
5. **Feature Engineering / Multi-Stream Setup (`normalize.py`)**: Calculates 3 additional streams:
   - **Bone Stream**: Computes `target_joint - source_joint`.
   - **Motion Stream**: Computes `joint(t+1) - joint(t)`.
   - **Bone Motion Stream**: Computes `bone(t+1) - bone(t)`.
6. **Temporal Resampling (`resample.py`)**: Because sign language videos have varying durations, all sequences are linearly interpolated to exactly 64 frames.

## Importance in Pipeline
Deep learning models are highly sensitive to noise. The spatial normalization removes confounding variables (camera distance, signer position), while the multi-stream feature engineering explicitly provides the model with velocity and angular information that it would otherwise struggle to learn from static coordinates alone.
