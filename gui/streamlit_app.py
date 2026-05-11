"""
Simple Streamlit GUI for single-video inference using the project's models.

Features:
- Upload a video file or pick a local path
- Choose model: LSTM, 3-stream ST-GCN, 4-stream (early/late)
- Run full MediaPipe extraction + preprocessing + inference
- Show predicted label and a tiny progress/status log

Run:
    pip install -r requirements.txt
    streamlit run gui/streamlit_app.py

Notes:
- This GUI calls the same codepath as `evaluation.predict_video`, so ensure
  `output/label_map.json` and model weights under `models/` exist if using
  pre-trained weights. For single-video runs the GUI will download MediaPipe
  models (if missing) and run extraction on the CPU by default.
"""

import streamlit as st
import os
import sys
import json
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Heavy ML imports are deferred until the user requests inference so
# Streamlit can start without installing torch/mediapipe.

OUTPUT_DIR = os.path.join(ROOT, "output")

st.set_page_config(page_title="Sign Language Demo", layout="wide")
st.title("Sign Language Recognition — Demo")

# Load label map (if present)
label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
if not os.path.exists(label_map_path):
    st.warning("label_map.json not found in output/. Run preprocessing/extract.py first to build the dataset and label map.")
    label_map = {}
else:
    with open(label_map_path) as f:
        label_map = json.load(f)


def _to_graph(x):
    """Reshape (T, F) -> (2, T, V) for ST-GCN inputs."""
    T, F = x.shape
    V = F // 2
    return x.reshape(T, V, 2).transpose(2, 0, 1).astype('float32')


col1, col2 = st.columns([2, 1])

with col1:
    st.header("Input")
    input_source = st.radio(
        "Input source",
        ["Upload / Local video", "Dataset video", "Dataset sample"],
        index=0,
    )

    uploaded = None
    local_path = ""
    sample_choice = None
    dataset_video_choice = None

    if input_source == "Upload / Local video":
        uploaded = st.file_uploader("Upload a short sign-language clip (mp4, avi)", type=["mp4", "avi", "mov"]) 
        st.markdown("or provide a path to a local file on the machine where Streamlit runs:")
        local_path = st.text_input("Local video path", value="")
    elif input_source == "Dataset sample":
        st.markdown("Select a sample from the dataset saved in `output/` (preprocessed streams).")
        y_path = os.path.join(OUTPUT_DIR, "y.npy")
        if not os.path.exists(y_path):
            st.warning("Dataset not found in output/. Run preprocessing to build the dataset first.")
        else:
            import numpy as np
            y = np.load(y_path)
            idx_to_label = {v: k for k, v in label_map.items()} if label_map else {}
            options = [f"{i} — {idx_to_label.get(int(lbl), str(int(lbl)))}" for i, lbl in enumerate(y)]
            sample_choice = st.selectbox("Choose sample index", options=options)
    else:
        st.markdown("Select a raw video from `dataset/` (organized by class folder).")
        dataset_root = os.path.join(ROOT, "dataset")
        if not os.path.exists(dataset_root):
            st.warning("dataset/ folder not found. Add your raw videos under dataset/<class_name>/.")
        else:
            video_exts = (".mp4", ".avi", ".mov")
            video_paths = []
            for root, _, files in os.walk(dataset_root):
                for name in files:
                    if name.lower().endswith(video_exts):
                        full = os.path.join(root, name)
                        rel = os.path.relpath(full, dataset_root)
                        video_paths.append((rel, full))
            video_paths.sort(key=lambda x: x[0])
            if not video_paths:
                st.warning("No videos found under dataset/. Add videos like dataset/<class>/video.mp4")
            else:
                labels = [p[0] for p in video_paths]
                selection = st.selectbox("Choose dataset video", options=labels)
                dataset_video_choice = dict(video_paths).get(selection)

with col2:
    st.header("Model")
    model_type = st.selectbox("Choose model to run", ["lstm", "3stream", "4stream-early", "4stream-late"]) 
    run_btn = st.button("Run Inference")

status = st.empty()
log = st.empty()


if run_btn:
    if not label_map:
        status.error("No label map available. Cannot run inference.")
        st.stop()

    # Dataset sample mode
    if input_source == "Dataset sample":
        if sample_choice is None:
            status.error("No dataset sample selected.")
            st.stop()

        idx = int(sample_choice.split(" ")[0])

        status.info("Loading model weights (may take a few seconds)...")
        try:
            from evaluation.evaluate import load_model
        except Exception as e:
            status.error(f"Failed to import model utilities: {e}\n\nInstall the project's dependencies (see requirements.txt) before running inference.")
            st.stop()

        model = load_model(len(label_map), model_type)

        status.info("Loading dataset streams and running inference...")
        try:
            import numpy as np

            if model_type == "lstm":
                for fname in ["X_final.npy", "X_normalized.npy", "X_raw.npy"]:
                    p = os.path.join(OUTPUT_DIR, fname)
                    if os.path.exists(p):
                        X = np.load(p)
                        break
                else:
                    status.error("No processed X_* data found in output/. Run preprocessing first.")
                    st.stop()

                sample = X[idx]
                try:
                    import torch
                    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out = model(tensor)
                        pred_idx = out.argmax(dim=1).item()
                except Exception as e:
                    status.error(f"Inference failed: {e}")
                    st.stop()

            else:
                try:
                    X_j  = np.load(os.path.join(OUTPUT_DIR, "X_normalized.npy"))
                    X_b  = np.load(os.path.join(OUTPUT_DIR, "X_bones.npy"))
                    X_m  = np.load(os.path.join(OUTPUT_DIR, "X_motion.npy"))
                    X_bm = np.load(os.path.join(OUTPUT_DIR, "X_bone_motion.npy"))
                except Exception as e:
                    status.error(f"Missing stream files in output/: {e}. Run normalize.py first.")
                    st.stop()

                def z_score(arr):
                    mean = arr.mean(axis=(0,1), keepdims=True)
                    std  = arr.std(axis=(0,1), keepdims=True)
                    std  = np.where(std < 1e-6, 1.0, std)
                    return (arr - mean) / std

                X_j = z_score(X_j)
                X_b = z_score(X_b)
                X_m = z_score(X_m)
                X_bm = z_score(X_bm)

                t_joint  = np.expand_dims(_to_graph(X_j[idx]), 0)
                t_bone   = np.expand_dims(_to_graph(X_b[idx]), 0)
                t_motion = np.expand_dims(_to_graph(X_m[idx]), 0)
                t_bm     = np.expand_dims(_to_graph(X_bm[idx]), 0)

                try:
                    import torch
                    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    tj = torch.tensor(t_joint, dtype=torch.float32).to(DEVICE)
                    tb = torch.tensor(t_bone, dtype=torch.float32).to(DEVICE)
                    tm = torch.tensor(t_motion, dtype=torch.float32).to(DEVICE)
                    tbm = torch.tensor(t_bm, dtype=torch.float32).to(DEVICE)

                    with torch.no_grad():
                        if model_type == "3stream":
                            out = model(tj, motion=tm, bone=tb)
                        else:
                            out = model(tj, motion=tm, bone=tb, bone_motion=tbm)
                        pred_idx = out.argmax(dim=1).item()
                except Exception as e:
                    status.error(f"Inference failed: {e}")
                    st.stop()

            idx_to_label = {v: k for k, v in label_map.items()}
            pred_label = idx_to_label.get(pred_idx, f"unknown({pred_idx})")
            status.success("Inference complete")
            st.subheader("Predicted label")
            st.write(f"**{pred_label}**")

        except Exception as e:
            status.error(f"Inference failed: {e}")

    else:
        # Upload / Local video branch
        # Prepare video file
        if input_source == "Dataset video":
            video_path = dataset_video_choice
            if not video_path:
                status.warning("Please select a video from the dataset.")
        elif uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
            tmp.write(uploaded.getbuffer())
            tmp.flush()
            video_path = tmp.name
        elif local_path:
            video_path = local_path
            if not os.path.exists(video_path):
                status.error(f"Local path not found: {video_path}")
                video_path = None
        else:
            status.warning("Please upload a video or provide a local path.")
            video_path = None

        if video_path:
            status.info("Loading model weights (may take a few seconds)...")
            try:
                from evaluation.evaluate import load_model, predict_video
            except Exception as e:
                status.error(f"Failed to import model utilities: {e}\n\nInstall the project's dependencies (see requirements.txt) before running inference.")
                st.stop()

            status.info("Running preprocessing and inference...")
            try:
                model = load_model(len(label_map), model_type)
                pred = predict_video(video_path, model, model_type, label_map)
                status.success("Inference complete")
                st.subheader("Predicted label")
                st.write(f"**{pred}**")
            except Exception as e:
                status.error(f"Inference failed: {e}")

            if uploaded is not None:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass


st.markdown("---")
st.caption("This demo runs the full pipeline locally. For faster real-time demos, consider exporting a lightweight pose detector or running on a machine with CUDA and switching to the YOLO backend.")
