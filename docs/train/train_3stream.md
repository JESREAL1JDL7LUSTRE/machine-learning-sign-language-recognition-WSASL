# train_3stream.py (3-Stream ST-GCN)

This document explains how the 3-stream training script works, what it expects, and what it produces.

## Purpose
The script trains a ThreeStream ST-GCN model using three synchronized skeleton streams:
- joint: normalized joint coordinates
- motion: frame-to-frame joint deltas
- bone: bone vectors (child minus parent)

The goal is to classify sign classes from skeleton sequences.

## Inputs and outputs
Inputs (from output/):
- X_final.npy or X_normalized.npy or X_raw.npy (joint stream)
- X_motion.npy (joint motion)
- X_bones.npy (bone vectors)
- y.npy (class labels)
- label_map.json (optional, for saving results)

Outputs:
- models/sign_stgcn_3stream.pth (best model weights)
- output/results_3stream.json (fold metrics, predictions, label map)
- output/X_test.npy, output/y_test.npy (held-out test split)

## Key configuration
- BATCH_SIZE = 4
- EPOCHS = 300
- LEARNING_RATE = 0.0005
- DROPOUT = 0.4
- K_FOLDS = 4
- TEST_SPLIT = 0.15
- PATIENCE = 35
- SWA_START_EP = 50, SWA_LR = 0.0001
- GRAPH_ARGS = {"layout": "mediapipe_51", "strategy": "spatial"}
- ADAPTIVE_GRAPH = True
- DROP_GRAPH_PROB = 0.1
- EARLY_FUSION = True
- LABEL_SMOOTHING = 0.1
- USE_WEIGHTED_LOSS = True

## Data format and shapes
Each stream uses the same time and joint layout:
- Original array shape: (N, T, F)
- F = 102 because there are 51 joints and 2 coordinates per joint
- The model expects graph tensors with shape (2, T, 51)

The dataset converts each sample by reshaping (T, F) into (2, T, V) where V = 51.

## Major components
### LabelSmoothingLoss
- Wraps cross-entropy with label smoothing and optional class weights.
- Smoothing spreads a small amount of probability mass to non-target classes.
- When `USE_WEIGHTED_LOSS` is True, each sample is scaled by its class weight.

### compute_class_weights
- Computes inverse-frequency weights from labels.
- Normalizes weights to have mean 1.

### augment_streams
Applies several stochastic augmentations to all three streams:
- Add Gaussian noise
- Global scaling
- Rotation in the 2D plane
- Horizontal flip (negate x)
- Temporal mask (drop frames and resample)
- Time warp (nonlinear index sampling)
- Random temporal crop + resample
- Random affine transforms over time segments

These are applied during training only.

### ThreeStreamDataset
- Stores three streams and labels.
- `to_graph` reshapes (T, F) to (2, T, 51).
- `__getitem__` returns three tensors plus the label.

### global_normalize
- Computes mean and std from the training split only.
- Applies the same stats to val and test splits.
- Avoids division by values < 1e-6.

### load_data
- Picks the first available joint stream in priority order:
  X_final.npy, X_normalized.npy, X_raw.npy
- Loads motion and bone arrays; missing files are replaced with zeros.
- Prints dataset stats (shape, class counts).

### eval_model
- Runs model in eval mode and returns accuracy and predictions.

## Training flow
1. Load joint, motion, bone, and labels.
2. Split data into train+val and a held-out test set (stratified).
3. Save the test split to output/ for later reuse.
4. Run K-fold CV on train+val:
   - For each fold, normalize streams using only the fold train split.
   - Train with early stopping and SWA accumulation.
   - Track best validation accuracy and store fold predictions.
5. After CV, load the best fold weights and evaluate on the held-out test set.
6. Save model weights and JSON results.

## Training loop details
- Optimizer: AdamW (weight_decay = 5e-2)
- LR schedule: cosine decay with warmup (LambdaLR)
- Gradient clipping: max_norm = 1.0
- Early stopping: stops after PATIENCE epochs without validation improvement

## SWA behavior
- SWA starts at `SWA_START_EP`.
- The script accumulates SWA weights without resetting BN each epoch.
- After training ends, it runs `update_bn` once and evaluates SWA.
- If SWA outperforms the base model, SWA weights are kept.

## Results JSON
The results file contains:
- fold_accs, cv_mean, cv_std
- test_acc
- all_preds, all_labels
- cv_preds, cv_labels
- num_classes
- fold_histories (train/val accuracy per epoch)
- label_map

## How to run
From the repo root:
- python training/train_3stream.py

The script will use GPU if available; otherwise CPU.
