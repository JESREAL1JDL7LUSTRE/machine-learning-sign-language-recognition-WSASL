**Step 1 — Check what you downloaded**
Your `data/` folder should now have: see link https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed?resource=download
```
data/
   videos/              ← folder with all the .mp4 files
   WLASL_v0.3.json      ← the label map file
   organize_dataset.py  ← the script we wrote
```

**Step 2 — Activate your venv**
```bash
cd "E:\SCHOOL\3rd year\ML"
venv\Scripts\activate
```

**Step 3 — Organize videos into class folders**
```bash
python data/organize_dataset.py --subset 100
```
This will create `dataset/hello/`, `dataset/book/`, etc. with the videos sorted inside. Using `--subset 100` keeps it manageable for a school project.

**Step 4 — Extract skeletons with MediaPipe**
```bash
# MediaPipe CPU (best quality, default)
python preprocessing/extract.py
python preprocessing/extract.py 20

# YOLO CPU
python preprocessing/extract.py 20 --backend yolo
python preprocessing/extract.py 20 --backend yolo --device cpu

# YOLO GPU (fastest)
python preprocessing/extract.py 20 --backend yolo --device cuda
python preprocessing/extract.py --device cuda        # auto-switches to yolo+cuda

# Custom frames
python preprocessing/extract.py 20 --frames 32
```
⚠️ This is the **slowest step** — it processes every video through MediaPipe. With 100 classes it could take **30–60 minutes** depending on your machine.

**Step 5 — Normalize**
```bash
python preprocessing/normalize.py
```

**Step 6 — Resample**
```bash
python preprocessing/resample.py
```

**Step 7 — Train**
```bash
python training/train.py

---
Additional `main.py` commands
```bash
# Retrain all models (ignore cache)
python main.py --compare-5 --force

# Quick smoke test (2 epochs)
python main.py --compare-5 --epochs 2 --force

# Retrain a single model
python main.py --bilstm --force

# Generate charts only from cached results
python main.py --compare-5 --results-only

# Skip chart generation (headless)
python main.py --compare-5 --no-charts

# Force device
python main.py --compare-5 --device cpu
python main.py --compare-5 --device cuda

# Remove cached results (PowerShell)
Remove-Item output\model_results.json

# Remove cached results (CMD)
del output\model_results.json
```
```