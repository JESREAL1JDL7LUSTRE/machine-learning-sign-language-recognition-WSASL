# training/

This directory contains the **training pipeline** for the four-stream ST-GCN sign language recognition model. It includes the dataset loader with augmentations, the full training loop with k-fold cross-validation, SWA, and early stopping.

---

## Files

| File | Description |
|---|---|
| `dataset.py` | PyTorch Dataset class with data augmentation for skeleton sequences |
| `train.py` | Full training script for the four-stream ST-GCN model |
| `train.txt` | Training log output (last run) |

---

## dataset.py — Skeleton Dataset with Augmentation

**What it does:**  
Provides two components used during training:

### Augmentation Functions

Applied independently to each sample during training (not at validation/test time):

| Function | Description | Probability |
|---|---|---|
| `add_noise(skeleton, noise_level=0.01)` | Gaussian jitter to all keypoints — simulates detector noise | 50% |
| `horizontal_flip(skeleton)` | Negate all x-coordinates — simulates opposite-handed signing | 50% |
| `random_frame_drop(skeleton, drop_prob=0.1)` | Randomly remove ~10% of frames then resample back | 30% |
| `time_warp(skeleton, sigma=0.15)` | Smooth random speed variation along time axis | 30% |

### SignDataset Class

```python
dataset = SignDataset(X, y, augment=True)
# X: shape (N, T, F) — skeleton features
# y: shape (N,) — integer class labels
# augment: True for training, False for validation/test
```

- `__getitem__` returns `(Tensor[T, F], long_label)`.
- Augmentations are applied independently each time a sample is accessed.
- The dataset is used by the simple `SignLSTM` (TCN) model; the ST-GCN trainer uses its own `FourStreamDataset` with a heavier augmentation pipeline.

---

## train.py — Four-Stream ST-GCN Training Script

**What it does:**  
Implements the full training pipeline for the `FourStreamSTGCN` model (four-stream early fusion ST-GCN). This is the primary standalone training script for the best-performing model in the project.

### Key Features

- **K-Fold Cross-Validation** (K=4): Provides robust performance estimates on small datasets.
- **Stratified train/test split** (15% held-out test set): Fixed split before cross-validation to prevent leakage.
- **Stochastic Weight Averaging (SWA)**: Accumulates model weight snapshots after epoch 50 and averages them, improving generalization over the final checkpoint.
- **Early stopping** (patience=35): Stops training when validation accuracy has not improved for 35 epochs.
- **Warmup-cosine LR schedule**: Linear warmup over 10 epochs, then cosine annealing.
- **Label smoothing** (ε=0.1): Softens one-hot targets to reduce overconfidence.
- **Class-weighted loss**: Compensates for class imbalance in small datasets.
- **Gradient clipping** (max norm=1.0): Prevents gradient explosions.
- **Heavy augmentation**: Rotation, scaling, translation, flipping, frame drop, time warp.

### Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `BATCH_SIZE` | 4 | Small batch for tiny datasets |
| `EPOCHS` | 300 | Max epochs per fold |
| `LEARNING_RATE` | 0.0005 | AdamW initial LR |
| `DROPOUT` | 0.4 | Dropout in model and fusion FC |
| `K_FOLDS` | 4 | Number of CV folds |
| `TEST_SPLIT` | 0.15 | Held-out test fraction |
| `PATIENCE` | 35 | Early stopping patience (epochs) |
| `SWA_START_EP` | 50 | Epoch to start accumulating SWA weights |
| `SWA_LR` | 0.0001 | SWA scheduler learning rate |
| `LABEL_SMOOTHING` | 0.1 | Label smoothing factor ε |
| `ADAPTIVE_GRAPH` | True | Enable learned adjacency correction |
| `DROP_GRAPH_PROB` | 0.1 | DropGraph probability |
| `EARLY_FUSION` | True | Use early fusion (vs. late fusion) |

### Key Components

#### `augment_skeleton(x)`

Heavy augmentation applied to all four input streams independently:
- **Gaussian noise** (60%): σ=0.015 in normalized space
- **Global scale jitter** (50%): random scale in [0.8, 1.2]
- **Global rotation** (50%): random angle in [-20°, +20°]
- **Horizontal flip** (50%): negate all x-coordinates
- **Frame drop** (40%): randomly drop ~15% of frames, resample back
- **Time warp** (40%): non-uniform frame sampling
- **Temporal crop** (40%): take a random 75–100% contiguous crop
- **Random move** (35%): smooth interpolated rotation + scale + translation

#### `FourStreamDataset`

Custom Dataset that holds all four data streams:
```python
# x_joint, x_motion, x_bone, x_bone_motion: each (N, T, 102)
dataset = FourStreamDataset(x_joint, x_motion, x_bone, x_bone_motion, y, augment=True)
```

`__getitem__` converts each stream from `(T, F)` to `(2, T, V)` graph format:  
```python
# (T, F=102) → (T, V=51, 2) → transpose → (2, T, 51)
x.reshape(T, V, 2).transpose(2, 0, 1)
```

#### `global_normalize(X_tr, *others)`

Z-score normalization using the training split statistics:
```python
X_tr_norm, X_val_norm, X_test_norm = global_normalize(X_tr, X_val, X_test)
```
Applied per-fold so the normalization stats are always computed from the training data only (prevents data leakage).

#### `LabelSmoothingLoss`

Custom loss combining label smoothing with optional per-class weights:
```
smoothed_target[y] = 1 - ε
smoothed_target[k≠y] = ε / (K - 1)
loss = -Σ smoothed_target * log_softmax(pred)
```

#### `train_fold(...)`

Runs one complete fold:
1. Creates train/val DataLoaders.
2. Builds a fresh `FourStreamSTGCN` model.
3. Trains for up to `EPOCHS` with early stopping.
4. After early stopping, evaluates SWA model (if ≥5 SWA snapshots).
5. Returns `(best_val_acc, best_state_dict)`.

#### `train()` — Main Entry Point

```
1. Load all four data streams
2. Hold out 15% test set (stratified)
3. For each of K=4 folds:
   a. Normalize per-fold (train stats)
   b. train_fold() → val_acc, state
4. Select best fold's weights
5. Evaluate best model on held-out test set
6. Save model to models/sign_stgcn.pth
```

### Usage

```bash
# Train the four-stream ST-GCN (standalone)
python training/train.py
```

### Output

| File | Description |
|---|---|
| `models/sign_stgcn.pth` | Best model weights (from best CV fold) |
| Console output | Per-epoch train/val accuracy, fold summaries, final test accuracy |

### SWA Details

**Stochastic Weight Averaging** runs as follows:
- From `SWA_START_EP=50` onwards, `swa_model.update_parameters(model)` averages the current model weights into a running average.
- After early stopping, `update_bn(train_loader, swa_model)` recomputes batch normalization statistics using one full pass over the training data (required because BN stats shift when weights are averaged).
- If the SWA model beats the best single checkpoint on validation, SWA weights are used instead.

---

## Data Format for Training

The trainer expects:
```
output/
├── X_final.npy         (N, 64, 102)  — joint stream (required)
├── X_bones.npy         (N, 64, 102)  — bone stream
├── X_motion.npy        (N, 64, 102)  — motion stream
├── X_bone_motion.npy   (N, 64, 102)  — bone motion stream
└── y.npy               (N,)          — class labels
```
If `X_bones.npy`, `X_motion.npy`, or `X_bone_motion.npy` are missing, zeros are used (model trains on a subset of streams with degraded performance).