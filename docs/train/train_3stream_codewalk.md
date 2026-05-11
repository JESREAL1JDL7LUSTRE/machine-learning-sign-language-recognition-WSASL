# train_3stream.py code walk (3-stream)

This is a code-focused walkthrough of how the script runs, in the same order the code is structured, plus a call-flow map.

## Call flow overview (high level)
1. Module constants and configs are defined.
2. Helper classes and functions are defined.
3. `train()` runs:
   - loads arrays
   - makes a stratified test split
   - performs K-fold CV on train+val
   - selects the best fold model
   - evaluates on the held-out test set
   - writes weights and results JSON
4. `if __name__ == "__main__"`: seeds are set, `train()` is called.

## Code map (what each block does)
- Global constants: configure training, SWA, graph layout, and loss behavior.
- `LabelSmoothingLoss`: cross-entropy with label smoothing and optional class weights.
- `compute_class_weights`: computes inverse-frequency weights to balance classes.
- `augment_streams`: heavy time+space augmentation applied to all 3 streams.
- `ThreeStreamDataset`: converts flat (T, F) to graph format (2, T, V) and returns tensors.
- `global_normalize`: fits mean/std on train split, applies to val/test without leakage.
- `load_data`: loads joint, motion, bone streams; prints dataset stats.
- `eval_model`: runs inference on a loader and returns accuracy and predictions.
- `train_fold`: trains one fold with early stopping + SWA.
- `train`: orchestrates the CV split, test evaluation, and saves outputs.

## Walkthrough by code order
### 1) Configuration and imports
The script imports numpy, torch, sklearn, and SWA utilities. Global constants define batch size, epochs, learning rate, dropout, fold count, and early stopping. It also defines graph layout and whether to use weighted loss.

The model class is imported as:
- `from models.st_gcn_twostream import Model as ThreeStreamSTGCN`
This model accepts multiple streams via keyword args (`motion`, `bone`).

### 2) Label smoothing loss
`LabelSmoothingLoss` wraps a log-softmax and constructs a smoothed target distribution:
- True class gets $1 - \epsilon$.
- All other classes get $\epsilon / (K - 1)$.
If class weights are enabled, each sample's loss is scaled by its class weight.

Why this matters:
- Prevents overconfident logits.
- Weighted loss combats class imbalance.

### 3) Class weighting
`compute_class_weights(y, num_classes, device)`:
- counts occurrences per class
- uses inverse frequency ($1 / count$)
- normalizes weights to mean 1
- returns a tensor on the target device

### 4) Data augmentation
`augment_streams(xj, xm, xb)` applies the same spatial/temporal transform to all streams:
- Noise: small Gaussian jitter
- Scale: uniform scale
- Rotation: planar rotation
- Flip: mirror on x-axis
- Temporal mask: drop some frames then resample to length T
- Time warp: nonlinear index sampling
- Random crop: choose a segment then resample to T
- Affine motion: piecewise rotation/scale/translation over time

This keeps the three streams consistent while injecting variability.

### 5) Dataset wrapper
`ThreeStreamDataset`:
- stores numpy arrays and labels
- `to_graph` reshapes (T, F) into (2, T, V) by grouping x/y
- `__getitem__` applies augmentation (if enabled) and returns tensors

The model expects each stream as (2, T, 51) and a label tensor.

### 6) Global normalization
`global_normalize`:
- computes mean and std from only the training split (axis over samples and time)
- applies those stats to val and test
- protects against division by near-zero std

This avoids train/val/test leakage.

### 7) Loading arrays
`load_data()`:
- picks first available joint array: X_final.npy, X_normalized.npy, or X_raw.npy
- loads y.npy
- loads X_bones.npy and X_motion.npy; if missing, uses zeros
- prints dataset shape and class distribution

### 8) Evaluation helper
`eval_model(m, val_loader)`:
- runs model in eval mode without grad
- collects predictions and labels
- returns accuracy and full prediction lists

### 9) Training one fold
`train_fold(...)`:
- builds train and val DataLoaders
- creates the model with graph config and dropout
- builds SWA model
- sets criterion (label smoothing + optional weights)
- optimizer: AdamW with weight decay
- scheduler: cosine with warmup

Training loop:
- forward pass with three streams
- loss backward
- gradient clipping
- optimizer step
- compute training accuracy
- update LR (SWA or normal scheduler)
- evaluate on val every epoch
- keep best weights and manage early stopping

After training:
- if enough SWA updates, run `update_bn` once and evaluate SWA
- choose SWA weights only if they improve val accuracy

Returns best val accuracy and state dict for this fold.

### 10) Top-level train()
`train()` handles overall orchestration:
1. Load streams and labels
2. Count classes and print parameter count
3. Split indices into train+val and test (stratified)
4. Save test split to output/
5. K-fold loop over train+val:
   - normalize each stream with fold train stats
   - run `train_fold`
   - track best fold and store best test-normalized arrays
6. Create model and load best weights
7. Run test evaluation
8. Save model weights and JSON metrics

The results JSON includes:
- fold accuracies and mean/std
- test accuracy
- all predictions and labels
- CV predictions
- per-fold training histories
- label map

## Execution entry
At the bottom:
- seeds are set for numpy and torch
- `train()` is called

Run from repo root:
- python training/train_3stream.py

## Why early fusion is "true" here
The model is built from the multi-stream ST-GCN implementation, and the script sets `EARLY_FUSION = True` so the model fuses streams early inside the network. The actual fusion logic lives in the model class.
