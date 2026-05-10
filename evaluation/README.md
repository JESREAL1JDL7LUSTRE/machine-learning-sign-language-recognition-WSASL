[← Back to Main README](../README.md)

# evaluation/

This directory contains the **evaluation and inference script** for the trained sign language recognition models.

---

## Files

| File | Description |
|---|---|
| `evaluate.py` | Evaluate a trained model on the dataset, or run inference on a single video |

---

## evaluate.py — Model Evaluation and Inference

**What it does:**  
Provides two modes of operation:

1. **Dataset evaluation mode** (default): Loads the full preprocessed dataset, runs the model in inference mode, and prints a detailed classification report and confusion matrix.

2. **Single video inference mode** (`--video`): Takes one raw video file, runs the full preprocessing pipeline on it (extract → normalize → resample), and prints the predicted sign class.

---

### Configuration

| Constant | Value | Description |
|---|---|---|
| `DEVICE` | auto | `cuda` if GPU available, else `cpu` |
| `MODEL_PATH` | `models/sign_lstm.pth` | Path to the saved model weights |
| `OUTPUT_DIR` | `output/` | Directory containing processed `.npy` files |
| `TARGET_LEN` | 64 | Fixed sequence length expected by the model |

---

### Functions

#### `load_model(num_classes) → SignLSTM`

Instantiates the `SignLSTM` (TCN + Attention) model with the standard architecture and loads pre-trained weights from `models/sign_lstm.pth`.

```python
model = load_model(num_classes=20)
```

The model configuration used:
```python
SignLSTM(
    input_size=150,   # raw 150-d MediaPipe features
    hidden_size=128,
    num_layers=2,
    num_classes=num_classes,
    dropout=0.3,
    bidirectional=True  # API compat flag; model is actually a TCN
)
```

> **Note:** This evaluator is designed for the TCN/LSTM model, not the ST-GCN. To evaluate the ST-GCN, use `main.py` which has its own evaluation loop.

---

#### `predict_video(video_path, model, label_map) → str`

Runs the complete preprocessing pipeline on a single raw video, then predicts the sign class.

**Full pipeline:**
```
Raw video
    → extract_skeleton()     — MediaPipe keypoint extraction
    → clean_missing_keypoints() — interpolate missing joints
    → normalize_skeleton()   — center + scale + relative hands
    → temporal_resample()    — resize to TARGET_LEN frames
    → to PyTorch tensor
    → model.forward()
    → argmax → class label string
```

```python
label = predict_video("path/to/video.mp4", model, label_map)
print(f"Predicted: {label}")
```

---

#### `evaluate_dataset()`

Loads the pre-processed dataset from `output/` and evaluates the model on all samples.

**Data loading priority:**  
Tries to load (in order):
1. `X_final.npy` — post-normalize, post-resample (best quality)
2. `X_normalized.npy` — post-normalize, pre-resample
3. `X_raw.npy` — raw extracted features

**Output:**

Prints a full `sklearn` classification report with per-class precision, recall, F1-score:
```
── Classification Report ─────────────────────────────
              precision    recall  f1-score   support

       hello       0.85      0.80      0.82        20
       thank       0.78      0.83      0.80        18
       ...
    accuracy                           0.81       200

── Confusion Matrix ──────────────────────────────────
[[16  2  1  1]
 [ 1 15  2  2]
 ...]
```

---

### Usage

**Evaluate on the full dataset:**
```bash
python evaluation/evaluate.py
```

**Run inference on a single video:**
```bash
python evaluation/evaluate.py --video path/to/sign_video.mp4
```

---

### Prerequisites

Before running the evaluator, ensure:

1. The preprocessing pipeline has been run:
   ```bash
   python preprocessing/extract.py
   python preprocessing/normalize.py
   python preprocessing/resample.py
   ```

2. A trained model exists at `models/sign_lstm.pth` (produced by `training/train.py` for the TCN model, or via training and saving manually).

3. `output/label_map.json` exists (created by `extract.py`).

---

### Notes

- **For the ST-GCN model:** Use `main.py --results-only` to load cached results and regenerate evaluation charts, or run a model with `main.py --multi-stream-stgcn` etc. to retrain and evaluate.
- **For the TCN model:** Use `evaluate.py` directly.
- **On the confusion matrix:** The matrix printed by `evaluate.py` operates on the FULL dataset (no train/test split), so it reflects training performance. For unbiased test performance, use the held-out test set reported by `train.py`.