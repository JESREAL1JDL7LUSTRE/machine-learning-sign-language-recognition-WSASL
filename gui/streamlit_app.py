"""Streamlit GUI for dataset-sample model comparison.

Features:
- Select one or more samples from the preprocessed dataset in `output/`.
- Compare ST-GCN variants and view per-sample accuracy.

Run:
    python -m streamlit run gui/streamlit_app.py
"""

import streamlit as st
import os
import sys
import json
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "output")

st.set_page_config(page_title="Sign Language Demo", layout="wide")
st.title("Sign Language Recognition — Demo")

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
    st.header("Samples")
    st.markdown("Pick one or more samples from the preprocessed dataset in `output/`.")
    sample_choices = []
    selected_indices = []
    y_path = os.path.join(OUTPUT_DIR, "y.npy")
    if not os.path.exists(y_path):
        st.warning("Dataset not found in output/. Run preprocessing to build the dataset first.")
    else:
        import numpy as np
        y = np.load(y_path)
        idx_to_label = {v: k for k, v in label_map.items()} if label_map else {}
        options = [f"{i} — {idx_to_label.get(int(lbl), str(int(lbl)))}" for i, lbl in enumerate(y)]
        run_all_samples = st.checkbox("Run all samples", value=False)
        if run_all_samples:
            selected_indices = list(range(len(y)))
            st.caption(f"{len(selected_indices)} samples selected.")
        else:
            sample_choices = st.multiselect("Choose sample indices", options=options, default=options[:1])
            selected_indices = [int(s.split(" ")[0]) for s in sample_choices]

with col2:
    st.header("Model")
    compare_models = st.checkbox("Compare multiple models", value=False)
    model_labels = {
        "3stream": "Multi-Stream ST-GCN",
        "4stream-early": "Four-Stream Early Fusion",
        "4stream-late": "Four-Stream Late Fusion",
    }
    model_options = ["3stream", "4stream-early", "4stream-late"]
    if compare_models:
        model_type = st.multiselect(
            "Choose models to run",
            model_options,
            default=["4stream-early", "4stream-late", "3stream"],
        )
    else:
        model_type = st.selectbox("Choose model to run", model_options)
    run_btn = st.button("Run Inference")

status = st.empty()
log = st.empty()


if run_btn:
    if not label_map:
        status.error("No label map available. Cannot run inference.")
        st.stop()

    if compare_models:
        if not model_type:
            status.error("Please select at least one model to compare.")
            st.stop()
        model_list = list(model_type)
    else:
        model_list = [model_type]

    if not selected_indices:
        status.error("No dataset samples selected.")
        st.stop()

    true_labels = {}
    try:
        import numpy as np
        y = np.load(os.path.join(OUTPUT_DIR, "y.npy"))
        idx_to_label = {v: k for k, v in label_map.items()}
        for idx in selected_indices:
            true_idx = int(y[idx])
            true_labels[idx] = idx_to_label.get(true_idx, f"unknown({true_idx})")
    except Exception:
        true_labels = {}

    status.info("Loading model weights (may take a few seconds)...")
    try:
        from evaluation.evaluate import load_model
    except Exception as e:
        status.error(f"Failed to import model utilities: {e}\n\nInstall the project's dependencies (see requirements.txt) before running inference.")
        st.stop()

    models = {m: load_model(len(label_map), m) for m in model_list}

    status.info("Loading dataset streams and running inference...")
    try:
        import numpy as np

        results = []
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

        for idx in selected_indices:
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
                    for m in model_list:
                        if m == "3stream":
                            out = models[m](tj, motion=tm, bone=tb)
                        else:
                            out = models[m](tj, motion=tm, bone=tb, bone_motion=tbm)
                        pred_idx = out.argmax(dim=1).item()
                        results.append((idx, m, pred_idx))
            except Exception as e:
                status.error(f"Inference failed: {e}")
                st.stop()

        idx_to_label = {v: k for k, v in label_map.items()}
        status.success("Inference complete")
        import pandas as pd
        rows = []
        preds = []
        for idx, m, i in results:
            label = None if i is None else idx_to_label.get(i, f"unknown({i})")
            if label is not None:
                preds.append(label)
            true_label = true_labels.get(idx)
            correct = "—"
            if true_label is not None and label is not None:
                correct = "✅" if label == true_label else "❌"
            rows.append({
                "sample": idx,
                "model": model_labels.get(m, m),
                "predicted": label or "—",
                "correct": correct,
            })

        st.subheader("Model comparison")
        st.dataframe(pd.DataFrame(rows))

        labeled = [r for r in rows if r["predicted"] != "—" and r["correct"] != "—"]
        if labeled:
            correct_count = sum(1 for r in labeled if r["correct"] == "✅")
            st.markdown(
                f"**Accuracy across selected samples:** {correct_count}/{len(labeled)} "
                f"({correct_count/len(labeled):.0%})"
            )
        if preds:
            from collections import Counter
            counts = Counter(preds)
            majority_label, majority_count = counts.most_common(1)[0]
            agreement = majority_count / len(preds)
            st.markdown(
                f"**Majority vote:** {majority_label}  \n"
                f"**Agreement:** {majority_count}/{len(preds)} ({agreement:.0%})"
            )

    except Exception as e:
        status.error(f"Inference failed: {e}")


st.markdown("---")
st.caption("This demo runs the full pipeline locally. For faster real-time demos, consider exporting a lightweight pose detector or running on a machine with CUDA and switching to the YOLO backend.")


def _run_evaluation_script(model_name: str) -> None:
    """Run `evaluation/evaluate.py --model <model_name>` and stream output to the app."""
    status.info(f"Running evaluation for {model_name}...")
    log_placeholder = st.empty()
    out_lines = []
    eval_script = os.path.join(ROOT, "evaluation", "evaluate.py")
    cmd = [sys.executable, eval_script, "--model", model_name]
    try:
        env = os.environ.copy()
        # Ensure subprocess prints UTF-8 so emojis and warnings don't raise on Windows console
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as e:
        status.error(f"Failed to start evaluation: {e}")
        return

    try:
        # Stream lines as they arrive
        for line in proc.stdout:
            out_lines.append(line)
            # keep viewport limited to last ~200 lines to avoid huge memory
            display_text = "".join(out_lines[-200:])
            log_placeholder.code(display_text)
        ret = proc.wait()
        if ret == 0:
            status.success(f"Evaluation finished ({model_name}).")
        else:
            status.error(f"Evaluation exited with code {ret}.")
    except Exception as e:
        status.error(f"Error while running evaluation: {e}")


st.markdown("**Run evaluation script**")
col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("Eval: 3stream"):
        _run_evaluation_script("3stream")
with col_b:
    if st.button("Eval: 4stream-early"):
        _run_evaluation_script("4stream-early")
with col_c:
    if st.button("Eval: 4stream-late"):
        _run_evaluation_script("4stream-late")
