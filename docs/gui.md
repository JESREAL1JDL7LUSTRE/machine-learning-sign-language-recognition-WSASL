# GUI Demo (Streamlit)

The project includes a lightweight Streamlit GUI for comparing trained ST-GCN models on samples from the preprocessed dataset in `output/`.

## Prerequisites

- Python environment with project dependencies installed.
- Preprocessed dataset files in `output/`:
  - `label_map.json`
  - `y.npy`
  - `X_normalized.npy`
  - `X_bones.npy`
  - `X_motion.npy`
  - `X_bone_motion.npy`
- Trained model weights in `models/` for the model variants you want to run:
  - `sign_stgcn_3stream.pth`
  - `sign_stgcn_4stream_early.pth`
  - `sign_stgcn_4stream_late.pth`

## Run the GUI

```bash
streamlit run gui/streamlit_app.py
```

If Streamlit is not on your PATH, run:

```bash
python3 -m streamlit run gui/streamlit_app.py
```

## Using the GUI

1. **Choose samples**: Select one or more dataset sample indices from the multiselect list. Each option includes the sample index and its ground-truth label from `y.npy` and `label_map.json`. Enable **Run all samples** to evaluate every sample in `y.npy`.
2. **Choose model mode**:
   - Leave **Compare multiple models** unchecked to run one model.
   - Enable **Compare multiple models** to run several variants side by side.
3. **Choose model variants**: Select from:
   - `3stream` - Multi-Stream ST-GCN using joint, bone, and motion streams.
   - `4stream-early` - Four-stream early fusion using joint, bone, motion, and bone-motion streams.
   - `4stream-late` - Four-stream late fusion using the same four streams.
4. **Run Inference**: Click **Run Inference** to load the selected weights and evaluate the selected samples.

## Results Shown

After inference completes, the GUI displays:

- A table with sample index, model name, predicted label, and whether the prediction matches the ground-truth label.
- Accuracy across the selected labeled samples.
- A majority vote summary across all predictions, including agreement count and percentage.

## Notes

- The GUI uses `evaluation.evaluate.load_model` to load the same model variants used by the evaluation scripts.
- Inference runs directly on the precomputed arrays in `output/`; it does not upload videos, read raw dataset videos, or re-extract skeleton keypoints.
- The app z-score normalizes each stream before reshaping samples into ST-GCN graph input format.
- If any required `output/` stream file is missing, rerun the preprocessing pipeline before opening the GUI.
- If model loading fails, confirm that dependencies from `requirements.txt` are installed and the expected `.pth` weight files exist in `models/`.
