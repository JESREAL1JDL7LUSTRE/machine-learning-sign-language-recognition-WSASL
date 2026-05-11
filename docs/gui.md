# GUI Demo (Streamlit)

The project includes a lightweight Streamlit GUI for running single-video inference, dataset-video inference, and dataset-sample inference.

## Prerequisites

- Python environment with project dependencies installed.
- Preprocessed dataset in `output/` (at least `label_map.json` and `y.npy`).
- Raw dataset videos organized under `dataset/<class_name>/` if you want to use **Dataset video** mode.
- For dataset-sample inference with ST‑GCN models: `X_normalized.npy`, `X_bones.npy`, `X_motion.npy`, `X_bone_motion.npy` in `output/`.
- For video inference: model weights in `models/` (e.g., `sign_stgcn_4stream_early.pth`) and MediaPipe/YOLO dependencies installed.

## Run the GUI

```bash
streamlit run gui/streamlit_app.py
```

If Streamlit is not on your PATH, run:

```bash
python -m streamlit run gui/streamlit_app.py
```

## Using the GUI

1. **Input source**
   - **Upload / Local video**: Upload a short clip or provide a local path.
   - **Dataset video**: Pick a raw video from `dataset/` (organized by class folder).
   - **Dataset sample**: Select a sample index from the preprocessed dataset in `output/`.
2. **Model**: Choose `lstm`, `3stream`, `4stream-early`, or `4stream-late`.
3. **Run Inference**: The GUI shows the predicted label for the chosen input.

## Notes

- The GUI uses the same model loading and preprocessing logic as `evaluation/evaluate.py`.
- Dataset-sample inference runs directly on the precomputed arrays and does not re-extract skeletons.
- If you only want to verify the UI, you can run the GUI without the heavy ML dependencies installed, but inference will require them.
