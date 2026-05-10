# System Analysis: Output

## Overview
The `output` module and directory serve as the central repository for processed data, trained model weights, evaluation logs, and visualization artifacts.

## Processed Data Tensors
The preprocessing pipeline outputs the following files into the `output` directory:
- `X_raw.npy`: Original skeleton coordinates extracted directly from videos. Shape: `(N, 64, 150)`.
- `X_normalized.npy`: Position- and scale-invariant skeleton coordinates. Shape: `(N, 64, 102)`.
- `X_bones.npy`: Extracted bone vectors derived from relative joint positions. Shape: `(N, 64, 102)`.
- `X_motion.npy`: Frame-to-frame joint velocities. Shape: `(N, 64, 102)`.
- `X_bone_motion.npy`: Frame-to-frame bone angular velocities. Shape: `(N, 64, 102)`.

## Evaluation Artifacts
- `model_results.json`: A comprehensive JSON log tracking cross-validation accuracy, test accuracy, inference predictions, and per-epoch histories for each evaluated model (`multi-stream-stgcn`, `4stream-late-fusion`, `4stream-fusion`).
- `charts/`: Contains `matplotlib`-generated visualizations, including confusion matrices, fold-level accuracy box plots, per-class accuracy bar charts, and overall model comparison plots.

## Importance in Pipeline
The output directory effectively decouples the data extraction phase from the modeling phase. Because the models ingest the finalized `.npy` arrays, we can rapidly iterate on neural network architectures without having to re-run the expensive MediaPipe video parsing.
