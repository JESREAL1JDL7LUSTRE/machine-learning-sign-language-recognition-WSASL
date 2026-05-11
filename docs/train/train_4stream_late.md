# train_4stream_late.py (4-Stream Late Fusion ST-GCN)

This document explains the 4-stream late-fusion training script and how it differs from early fusion.

## Purpose
The script trains a FourStream ST-GCN model using four synchronized streams:
- joint: normalized joint coordinates
- motion: frame-to-frame joint deltas
- bone: bone vectors (child minus parent)
- bone_motion: frame-to-frame bone deltas

Late fusion means the model keeps streams more separate and fuses later in the network.

## Inputs and outputs
Inputs (from output/):
- X_final.npy or X_normalized.npy or X_raw.npy (joint stream)
- X_motion.npy (joint motion)
- X_bones.npy (bone vectors)
- X_bone_motion.npy (bone motion)
- y.npy (class labels)
- label_map.json (optional, for saving results)

Outputs:
- models/sign_stgcn_4stream_late.pth (best model weights)
- output/results_4stream_late.json (fold metrics, predictions, label map)
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
- EARLY_FUSION = False
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
- Cross-entropy with label smoothing and optional class weights.

### compute_class_weights
- Inverse-frequency class weights, normalized to mean 1.

### augment_4streams
Applies the same heavy augmentation to all four streams:
- Add Gaussian noise
- Global scaling
- Rotation in the 2D plane
- Horizontal flip (negate x)
- Temporal mask (drop frames and resample)
- Time warp (nonlinear index sampling)
- Random temporal crop + resample
- Random affine transforms over time segments

### FourStreamDataset
- Stores all four streams and labels.
- `to_graph` reshapes (T, F) to (2, T, 51).
- `__getitem__` returns four tensors plus the label.

### global_normalize
- Computes mean and std on the training split only, then applies to val and test.

### load_data
- Picks the first available joint stream in order:
  X_final.npy, X_normalized.npy, X_raw.npy
- Loads motion, bone, and bone_motion; missing files become zeros.

### eval_model
- Runs the model and returns accuracy and predictions.

## Training flow
1. Load four streams and labels.
2. Create a held-out test split (stratified).
3. Save the test split to output/.
4. Run K-fold CV on train+val:
   - Normalize per fold using only the training split.
   - Train with early stopping and SWA accumulation.
   - Track best validation accuracy and store fold predictions.
5. Use the best fold weights to evaluate the held-out test set.
6. Save model and JSON results.

## Training loop details
- Optimizer: AdamW (weight_decay = 5e-2)
- LR schedule: cosine decay with warmup (LambdaLR)
- Gradient clipping: max_norm = 1.0
- Early stopping: PATIENCE epochs without validation improvement

## SWA behavior
- SWA starts at `SWA_START_EP`.
- The script keeps a running SWA model and updates BN once at the end.
- If SWA beats the base model, SWA weights are used.

## Results JSON
The results file contains:
- fold_accs, cv_mean, cv_std
- test_acc
- all_preds, all_labels
- cv_preds, cv_labels
- num_classes
- fold_histories
- label_map

## How to run
From the repo root:
- python training/train_4stream_late.py
