"""
extract.py — Unified skeleton extractor for Sign Language Recognition
======================================================================

Supports two backends:
  • mediapipe  (CPU only, best quality — 150 features)
  • yolo       (GPU supported, faster — 118 features)

Usage examples:
  python preprocessing/extract.py                        # mediapipe, all classes
  python preprocessing/extract.py 20                     # mediapipe, first 20 classes
  python preprocessing/extract.py 20 --backend yolo      # yolo, first 20 classes
  python preprocessing/extract.py --device cuda          # yolo + GPU (auto-switches to yolo)
  python preprocessing/extract.py --device cpu           # mediapipe, CPU
  python preprocessing/extract.py 20 --backend yolo --device cuda
  python preprocessing/extract.py --frames 32            # custom frame count
"""

import cv2
import numpy as np
import os
import json
import argparse
import urllib.request
import shutil
import torch
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Feature sizes ─────────────────────────────────────────────────────────────
# MediaPipe: Pose(33*2=66) + LeftHand(21*2=42) + RightHand(21*2=42) = 150
# YOLO     : Body(17*2=34) + LeftHand(21*2=42) + RightHand(21*2=42) = 118

MP_FEATURE_SIZE   = 150
YOLO_FEATURE_SIZE = 118
MAX_FRAMES        = 64

# ── Model paths ───────────────────────────────────────────────────────────────
MP_MODELS_DIR   = os.path.join(ROOT, "data", "mp_models")
YOLO_MODELS_DIR = os.path.join(ROOT, "data", "yolo_models")

POSE_MODEL_PATH = os.path.join(MP_MODELS_DIR, "pose_landmarker_full.task")
HAND_MODEL_PATH = os.path.join(MP_MODELS_DIR, "hand_landmarker.task")
BODY_MODEL_PATH = os.path.join(YOLO_MODELS_DIR, "yolov8n-pose.pt")
YOLO_HAND_PATH  = os.path.join(YOLO_MODELS_DIR, "yolo11n-pose-hands.pt")

POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
YOLO_HAND_URL   = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE UTILS
# ══════════════════════════════════════════════════════════════════════════════

def resolve_device(preferred=None, backend=None):
    """
    Resolve device and backend together.

    Rules:
      --device cuda  → forces YOLO backend (MediaPipe has no GPU on Windows)
      --device cpu   → uses whichever backend is selected
      auto           → if CUDA available and backend=yolo, use cuda; else cpu
    """
    cuda_available = torch.cuda.is_available()

    if preferred == 'cuda':
        if not cuda_available:
            raise RuntimeError(
                "❌ CUDA requested but not available.\n"
                "   pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu121"
            )
        if backend == 'mediapipe':
            print("  ⚠️  MediaPipe has no GPU support on Windows.")
            print("  ⚠️  Switching backend to YOLO for CUDA support.")
            backend = 'yolo'
        return 'cuda', backend

    if preferred == 'cpu':
        return 'cpu', backend

    # Auto-detect
    if cuda_available and backend == 'yolo':
        print(f"  🚀 GPU detected: {torch.cuda.get_device_name(0)} — using CUDA")
        return 'cuda', backend

    return 'cpu', backend


# ══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def download_mp_models():
    """Download MediaPipe task model files if not present."""
    os.makedirs(MP_MODELS_DIR, exist_ok=True)
    for path, url, name in [
        (POSE_MODEL_PATH, POSE_MODEL_URL, "MP Pose model"),
        (HAND_MODEL_PATH, HAND_MODEL_URL, "MP Hand model"),
    ]:
        if not os.path.exists(path):
            print(f"  Downloading {name}...")
            urllib.request.urlretrieve(url, path)
            print(f"  ✅ {name} saved")
        else:
            print(f"  ✅ {name} already exists")


def make_mp_detectors():
    """Create MediaPipe pose and hand detectors."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, PoseLandmarkerOptions,
        HandLandmarker, HandLandmarkerOptions
    )
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

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
    return (
        PoseLandmarker.create_from_options(pose_options),
        HandLandmarker.create_from_options(hand_options)
    )


def extract_mp(video_path, pose_det, hand_det, max_frames=MAX_FRAMES):
    """
    Extract skeleton using MediaPipe Tasks API.
    Returns: (max_frames, 150)
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    keypoints = []

    while cap.isOpened() and len(keypoints) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_kp = []

        # Pose (66)
        pose_res = pose_det.detect(img)
        if pose_res.pose_landmarks:
            for lm in pose_res.pose_landmarks[0]:
                frame_kp.extend([lm.x, lm.y])
        else:
            frame_kp.extend([0.0] * 66)

        # Hands (84)
        left_kp  = [0.0] * 42
        right_kp = [0.0] * 42
        hand_res = hand_det.detect(img)
        if hand_res.hand_landmarks:
            for i, hand_lms in enumerate(hand_res.hand_landmarks):
                if i >= len(hand_res.handedness):
                    break
                side = hand_res.handedness[i][0].display_name
                kp   = [v for lm in hand_lms for v in [lm.x, lm.y]]
                if side == "Left":
                    left_kp = kp
                else:
                    right_kp = kp

        frame_kp.extend(left_kp)
        frame_kp.extend(right_kp)
        keypoints.append(frame_kp)

    cap.release()

    while len(keypoints) < max_frames:
        keypoints.append([0.0] * MP_FEATURE_SIZE)

    return np.array(keypoints, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# YOLO BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def download_yolo_models():
    """Download YOLO body + hand models if not present."""
    from ultralytics import YOLO
    os.makedirs(YOLO_MODELS_DIR, exist_ok=True)

    if not os.path.exists(BODY_MODEL_PATH):
        print("  Downloading YOLO body model...")
        YOLO("yolov8n-pose.pt")
        if os.path.exists("yolov8n-pose.pt"):
            shutil.move("yolov8n-pose.pt", BODY_MODEL_PATH)
        print(f"  ✅ YOLO body model saved")
    else:
        print(f"  ✅ YOLO body model already exists")

    if not os.path.exists(YOLO_HAND_PATH):
        print("  Downloading YOLO hand model...")
        try:
            urllib.request.urlretrieve(YOLO_HAND_URL, YOLO_HAND_PATH)
            print(f"  ✅ YOLO hand model saved")
        except Exception as e:
            print(f"  ⚠️  Hand model download failed: {e}")
            print("     Hand keypoints will be zeros.")
    else:
        print(f"  ✅ YOLO hand model already exists")


def make_yolo_models():
    """Load YOLO body and hand models."""
    from ultralytics import YOLO
    body = YOLO(BODY_MODEL_PATH)
    hand = YOLO(YOLO_HAND_PATH) if os.path.exists(YOLO_HAND_PATH) else None
    return body, hand


def extract_yolo(video_path, body_model, hand_model, device='cpu', max_frames=MAX_FRAMES):
    """
    Extract skeleton using YOLO pose models.
    Returns: (max_frames, 118)
    """
    cap = cv2.VideoCapture(video_path)
    keypoints = []

    while cap.isOpened() and len(keypoints) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_kp = []

        # Body (34)
        body_res = body_model(frame, verbose=False, device=device)
        body_kp  = [0.0] * 34
        if body_res and body_res[0].keypoints is not None:
            kps = body_res[0].keypoints.xy
            if len(kps) > 0 and kps[0].shape[0] == 17:
                body_kp = kps[0].cpu().numpy().flatten().tolist()
        frame_kp.extend(body_kp)

        # Hands (84)
        left_kp  = [0.0] * 42
        right_kp = [0.0] * 42
        if hand_model is not None:
            hand_res = hand_model(frame, verbose=False, device=device)
            if hand_res and hand_res[0].keypoints is not None:
                detected = []
                for i in range(len(hand_res[0].keypoints)):
                    kps = hand_res[0].keypoints.xy[i].cpu().numpy()
                    if kps.shape[0] == 21:
                        detected.append((kps[0][0], kps.flatten().tolist()))
                detected.sort(key=lambda h: h[0])
                if len(detected) >= 1:
                    right_kp = detected[0][1]
                if len(detected) >= 2:
                    left_kp  = detected[1][1]

        frame_kp.extend(right_kp)
        frame_kp.extend(left_kp)
        keypoints.append(frame_kp)

    cap.release()

    while len(keypoints) < max_frames:
        keypoints.append([0.0] * YOLO_FEATURE_SIZE)

    return np.array(keypoints, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(data_dir, save_dir, backend='mediapipe', device='cpu',
                  max_frames=MAX_FRAMES, subset=None):

    os.makedirs(save_dir, exist_ok=True)
    feature_size = MP_FEATURE_SIZE if backend == 'mediapipe' else YOLO_FEATURE_SIZE

    # ── Setup backend ─────────────────────────────────────────────────────────
    print(f"\nSetting up {backend.upper()} backend...")
    if backend == 'mediapipe':
        download_mp_models()
        pose_det, hand_det = make_mp_detectors()
        extractfn = lambda vp: extract_mp(vp, pose_det, hand_det, max_frames)
    else:
        download_yolo_models()
        body_model, hand_model = make_yolo_models()
        extractfn = lambda vp: extract_yolo(vp, body_model, hand_model, device, max_frames)

    # ── Scan class folders ────────────────────────────────────────────────────
    EXCLUDE = {'videos', 'mp_models', 'yolo_models'}
    class_folders = sorted([
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
        and f.lower() not in EXCLUDE
    ])

    if subset:
        class_folders = class_folders[:subset]

    if len(class_folders) == 0:
        print(f"❌ No class folders found. Run organize_dataset.py first.")
        return None, None, None

    print(f"\n  Backend  : {backend.upper()}")
    print(f"  Device   : {device.upper()}")
    print(f"  Features : {feature_size} per frame")
    print(f"  Classes  : {len(class_folders)}")
    print(f"  Frames   : {max_frames} per video\n")

    label_map = {}
    data      = []
    labels    = []

    for idx, label in enumerate(class_folders):
        folder_path = os.path.join(data_dir, label)
        label_map[label] = idx

        video_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]

        print(f"[{idx+1}/{len(class_folders)}] '{label}' — {len(video_files)} videos")

        for vid_file in tqdm(video_files, leave=False):
            vid_path = os.path.join(folder_path, vid_file)
            try:
                skeleton = extractfn(vid_path)
                data.append(skeleton)
                labels.append(idx)
            except Exception as e:
                print(f"  [ERROR] Skipped {vid_file}: {e}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if backend == 'mediapipe':
        pose_det.close()
        hand_det.close()

    # ── Save ──────────────────────────────────────────────────────────────────
    X = np.array(data,   dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    np.save(os.path.join(save_dir, "X_raw.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"),     y)

    with open(os.path.join(save_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n✅ Done!")
    print(f"   Backend  : {backend.upper()}")
    print(f"   Device   : {device.upper()}")
    print(f"   X shape  : {X.shape}  (samples, frames, {feature_size} features)")
    print(f"   y shape  : {y.shape}")
    print(f"   Classes  : {len(label_map)}")

    return X, y, label_map


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract skeleton keypoints from sign language videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocessing/extract.py                         # mediapipe, all classes
  python preprocessing/extract.py 20                      # mediapipe, first 20 classes
  python preprocessing/extract.py 20 --backend yolo       # yolo, first 20 classes
  python preprocessing/extract.py --device cuda           # auto-switch to yolo+GPU
  python preprocessing/extract.py 20 --backend yolo --device cuda
        """
    )
    parser.add_argument(
        "subset", type=int, nargs="?", default=None,
        help="Only process first N classes (e.g. 20)"
    )
    parser.add_argument(
        "--backend", type=str, default="mediapipe",
        choices=["mediapipe", "yolo"],
        help="Extraction backend (default: mediapipe)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cuda", "cpu"],
        help="Device: cuda=GPU (yolo only), cpu=CPU. Default: auto"
    )
    parser.add_argument(
        "--frames", type=int, default=MAX_FRAMES,
        help=f"Max frames per video (default: {MAX_FRAMES})"
    )

    args = parser.parse_args()

    # Resolve device + backend together
    device, backend = resolve_device(args.device, args.backend)

    DATA_DIR = os.path.join(ROOT, "dataset")
    SAVE_DIR = os.path.join(ROOT, "output")

    build_dataset(
        DATA_DIR, SAVE_DIR,
        backend    = backend,
        device     = device,
        max_frames = args.frames,
        subset     = args.subset
    )