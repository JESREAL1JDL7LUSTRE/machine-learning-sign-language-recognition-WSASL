"""
run_top1000_classes.py — Full pipeline for the top-1000 classes
=============================================================================
Runs: extract -> normalize -> resample -> train (4-stream early fusion ONLY)
"""

import os
import sys
import json
import time
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

OUTPUT_DIR  = os.path.join(ROOT, "output")
DATASET_DIR = os.path.join(ROOT, "dataset")

# ── Dynamic top 1000 classes ──────────────────────────────────────────────────
classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)) and d != 'videos']
counts = {}
for c in classes:
    vids = [f for f in os.listdir(os.path.join(DATASET_DIR, c)) if f.endswith('.mp4')]
    counts[c] = len(vids)
sorted_classes = sorted(counts.items(), key=lambda x: x[1], reverse=True)
TOP_CLASSES = [c for c, count in sorted_classes[:1000]]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — EXTRACT KEYPOINTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"STEP 1/4 — Extracting skeleton keypoints for {len(TOP_CLASSES)} classes")
print("="*60)

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import urllib.request

MP_MODELS_DIR  = os.path.join(ROOT, "data", "mp_models")
POSE_MODEL_PATH = os.path.join(MP_MODELS_DIR, "pose_landmarker_full.task")
HAND_MODEL_PATH = os.path.join(MP_MODELS_DIR, "hand_landmarker.task")
os.makedirs(MP_MODELS_DIR, exist_ok=True)

def download_if_missing(path, url, name):
    if not os.path.exists(path):
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(url, path)
        print(f"  Done: {name}")

download_if_missing(POSE_MODEL_PATH, "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task", "MP Pose model")
download_if_missing(HAND_MODEL_PATH, "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task", "MP Hand model")

pose_options = PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=VisionTaskRunningMode.IMAGE,
    num_poses=1
)
hand_options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionTaskRunningMode.IMAGE,
    num_hands=2
)
pose_det = PoseLandmarker.create_from_options(pose_options)
hand_det = HandLandmarker.create_from_options(hand_options)

MAX_FRAMES   = 64
FEATURE_SIZE = 150

def extract_mp(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    while cap.isOpened() and len(keypoints) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_kp = []
        pose_res = pose_det.detect(img)
        if pose_res.pose_landmarks:
            for lm in pose_res.pose_landmarks[0]:
                frame_kp.extend([lm.x, lm.y])
        else:
            frame_kp.extend([0.0] * 66)
        left_kp  = [0.0] * 42
        right_kp = [0.0] * 42
        hand_res = hand_det.detect(img)
        if hand_res.hand_landmarks:
            for i, hand_lms in enumerate(hand_res.hand_landmarks):
                if i >= len(hand_res.handedness): break
                side = hand_res.handedness[i][0].display_name
                kp   = [v for lm in hand_lms for v in [lm.x, lm.y]]
                if side == "Left": left_kp = kp
                else: right_kp = kp
        frame_kp.extend(left_kp)
        frame_kp.extend(right_kp)
        keypoints.append(frame_kp)
    cap.release()
    while len(keypoints) < MAX_FRAMES:
        keypoints.append([0.0] * FEATURE_SIZE)
    return np.array(keypoints, dtype=np.float32)

data    = []
labels  = []
label_map = {}
t0 = time.time()

for idx, cls in enumerate(TOP_CLASSES):
    label_map[cls] = idx
    folder = os.path.join(DATASET_DIR, cls)
    videos = [f for f in os.listdir(folder) if f.lower().endswith('.mp4')]
    print(f"[{idx+1:4d}/{len(TOP_CLASSES)}] '{cls:<15}' — {len(videos):>2d} videos", end="", flush=True)
    ok = 0
    for vf in videos:
        vpath = os.path.join(folder, vf)
        try:
            skel = extract_mp(vpath)
            data.append(skel)
            labels.append(idx)
            ok += 1
        except Exception as e:
            pass
    print(f" -> Extracted {ok}")

pose_det.close()
hand_det.close()

X_raw = np.array(data,   dtype=np.float32)
y     = np.array(labels, dtype=np.int64)
np.save(os.path.join(OUTPUT_DIR, "X_raw.npy"), X_raw)
np.save(os.path.join(OUTPUT_DIR, "y.npy"),     y)
with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

elapsed = time.time() - t0
print(f"\nExtraction done in {elapsed/60:.1f} min")
print(f"X_raw shape: {X_raw.shape}  |  Classes: {len(label_map)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — NORMALIZE + COMPUTE 4 STREAMS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2/4 — Normalizing + computing 4 feature streams")
print("="*60)

from preprocessing.normalize import (
    filter_joints, clean_missing_keypoints,
    normalize_dataset, smooth_dataset,
    compute_bone_vectors, compute_motion
)

X_clean = np.array([clean_missing_keypoints(X_raw[i]) for i in range(len(X_raw))], dtype=np.float32)
X_filt  = filter_joints(X_clean)
X_norm  = normalize_dataset(X_filt)
X_smooth = smooth_dataset(X_norm, sigma=0.8)
X_bones  = compute_bone_vectors(X_smooth)
X_motion = compute_motion(X_smooth)
X_bm     = compute_motion(X_bones)

np.save(os.path.join(OUTPUT_DIR, "X_normalized.npy"),   X_smooth)
np.save(os.path.join(OUTPUT_DIR, "X_final.npy"),        X_smooth)
np.save(os.path.join(OUTPUT_DIR, "X_bones.npy"),        X_bones)
np.save(os.path.join(OUTPUT_DIR, "X_motion.npy"),       X_motion)
np.save(os.path.join(OUTPUT_DIR, "X_bone_motion.npy"),  X_bm)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN 4-STREAM EARLY FUSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3/4 — Training 4-Stream Early Fusion Model")
print("="*60)

import subprocess
# Only running the 4-stream early fusion model
subprocess.run([sys.executable, "main.py", "--4stream-fusion", "--force", "--epochs", "150"], cwd=ROOT)

print("\nAll done! Results saved to output/model_results.json and output/charts/")
