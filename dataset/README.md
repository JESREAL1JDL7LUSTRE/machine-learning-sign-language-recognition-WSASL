# Dataset Module

## Purpose
Handles loading and structuring of the WLASL dataset for training and testing.

## Responsibilities
- Read dataset annotations (JSON/CSV)
- Load raw data (videos, frames, or keypoints)
- Apply train/test split (70/30)
- Return data in a format usable by models

## Expected Output
Each sample should return:
- `x`: input features (frames or keypoints)
- `y`: label (gloss/class index)

## Notes
- Do NOT include heavy preprocessing here (e.g., pose extraction)
- Keep this module focused on data access only
- Ensure reproducibility of splits (store in /data/splits)

## Future Extensions
- Add support for batching
- Add caching for faster loading