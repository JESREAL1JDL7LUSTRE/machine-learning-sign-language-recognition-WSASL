"""
extract.py — Skeleton Keypoint Extraction from Sign Language Videos
====================================================================

This is the first step of the preprocessing pipeline. It takes raw video
files organized into class folders (one folder per sign word) and extracts
skeleton keypoints frame-by-frame using one of two computer vision backends:

Backend options:
    MediaPipe (default, CPU-only on Windows):
        - Pose landmark model: 33 upper-body keypoints × 2 (x, y) = 66 features
        - Hand landmark model: 21 keypoints × 2 per hand × 2 hands = 84 features
        - Total per frame: 66 + 84 = 150 features

    YOLO (GPU-capable):
        - YOLOv8n-pose body: 17 keypoints × 2 = 34 features
        - YOLO hand model:   21 keypoints × 2 per hand × 2 hands = 84 features
        - Total per frame: 34 + 84 = 118 features

Output files (saved to output/):
    X_raw.npy     — shape (N, max_frames, feature_size) — raw skeleton data
    y.npy         — shape (N,) — integer class labels
    label_map.json — dict mapping class_name → class_index

Usage:
    # All classes, MediaPipe backend, CPU
    python preprocessing/extract.py

    # First 20 classes only
    python preprocessing/extract.py 20

    # YOLO backend, GPU
    python preprocessing/extract.py --backend yolo --device cuda

    # MediaPipe, limit to 64 frames per video
    python preprocessing/extract.py --frames 64
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

# Build the absolute path to the project root from this file's location
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Feature sizes ─────────────────────────────────────────────────────────────
# MediaPipe: Pose(33 joints × 2 = 66) + LeftHand(21 × 2 = 42) + RightHand(21 × 2 = 42) = 150
# YOLO     : Body(17 joints × 2 = 34) + LeftHand(21 × 2 = 42) + RightHand(21 × 2 = 42) = 118

MP_FEATURE_SIZE   = 150   # total features per frame for MediaPipe backend
YOLO_FEATURE_SIZE = 118   # total features per frame for YOLO backend
MAX_FRAMES        = 64    # default: cap each video at 64 frames

# ── Model download paths ───────────────────────────────────────────────────────
# MediaPipe .task files are downloaded once from Google storage
MP_MODELS_DIR   = os.path.join(ROOT, "data", "mp_models")
YOLO_MODELS_DIR = os.path.join(ROOT, "data", "yolo_models")

POSE_MODEL_PATH = os.path.join(MP_MODELS_DIR, "pose_landmarker_full.task")
HAND_MODEL_PATH = os.path.join(MP_MODELS_DIR, "hand_landmarker.task")
BODY_MODEL_PATH = os.path.join(YOLO_MODELS_DIR, "yolov8n-pose.pt")
YOLO_HAND_PATH  = os.path.join(YOLO_MODELS_DIR, "yolo11n-pose-hands.pt")

# ── Download URLs ──────────────────────────────────────────────────────────────
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
YOLO_HAND_URL   = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE UTILS
# ══════════════════════════════════════════════════════════════════════════════

def resolve_device(preferred=None, backend=None):
    """Resolve the compute device and backend, handling MediaPipe's GPU limitation.

    MediaPipe does NOT support GPU on Windows. If the user requests CUDA with
    MediaPipe, this function automatically switches to the YOLO backend.

    Resolution rules:
        cuda + mediapipe → warns and switches to yolo + cuda
        cuda + yolo      → uses GPU
        cpu  + any       → uses CPU with the requested backend
        auto + yolo      → uses CUDA if available, else CPU
        auto + mediapipe → always CPU (MediaPipe cannot use GPU on Windows)

    Args:
        preferred (str or None): 'cuda', 'cpu', or None (auto-detect).
        backend   (str or None): 'mediapipe' or 'yolo'.

    Returns:
        Tuple[str, str]: (device_str, backend_str)

    Raises:
        RuntimeError: If 'cuda' requested but no CUDA device is available.
    """
    cuda_available = torch.cuda.is_available()

    if preferred == 'cuda':
        # User explicitly requested CUDA
        if not cuda_available:
            raise RuntimeError(
                "❌ CUDA requested but not available.\n"
                "   pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu121"
            )
        if backend == 'mediapipe':
            # MediaPipe cannot use GPU on Windows — switch to YOLO automatically
            print("  ⚠️  MediaPipe has no GPU support on Windows.")
            print("  ⚠️  Switching backend to YOLO for CUDA support.")
            backend = 'yolo'
        return 'cuda', backend

    if preferred == 'cpu':
        # User explicitly requested CPU — honour it regardless of GPU availability
        return 'cpu', backend

    # Auto-detect: use GPU only if YOLO backend is selected (MediaPipe is CPU-only)
    if cuda_available and backend == 'yolo':
        print(f"  🚀 GPU detected: {torch.cuda.get_device_name(0)} — using CUDA")
        return 'cuda', backend

    # Default: CPU
    return 'cpu', backend


# ══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def download_mp_models():
    """Download MediaPipe task model files if they are not already present.

    MediaPipe Tasks API requires pre-downloaded .task files.
    They are cached in data/mp_models/ after the first download.
    """
    os.makedirs(MP_MODELS_DIR, exist_ok=True)
    # Download both pose and hand models if missing
    for path, url, name in [
        (POSE_MODEL_PATH, POSE_MODEL_URL, "MP Pose model"),
        (HAND_MODEL_PATH, HAND_MODEL_URL, "MP Hand model"),
    ]:
        if not os.path.exists(path):
            print(f"  Downloading {name}...")
            urllib.request.urlretrieve(url, path)  # blocking download
            print(f"  ✅ {name} saved")
        else:
            print(f"  ✅ {name} already exists")


def make_mp_detectors():
    """Create MediaPipe pose and hand landmark detector instances.

    Uses the Tasks API (mediapipe.tasks.python.vision) with IMAGE running mode,
    which processes one frame at a time (no video streaming).

    Returns:
        Tuple[PoseLandmarker, HandLandmarker]: Initialized detector pair.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, PoseLandmarkerOptions,
        HandLandmarker, HandLandmarkerOptions
    )
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

    # Pose detector: detects up to 1 person's body landmarks per frame
    pose_options = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=VisionTaskRunningMode.IMAGE,  # single-image mode
        num_poses=1                                 # only track one signer
    )
    # Hand detector: detects up to 2 hands (left and right) per frame
    hand_options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=2    # both hands
    )
    return (
        PoseLandmarker.create_from_options(pose_options),
        HandLandmarker.create_from_options(hand_options)
    )


def extract_mp(video_path, pose_det, hand_det, max_frames=MAX_FRAMES):
    """Extract per-frame skeleton keypoints from a video using MediaPipe.

    For each frame:
        - Runs PoseLandmarker: extracts 33 landmarks × 2 = 66 features
          (only (x, y) — depth z is discarded for 2-D graph compatibility)
        - Runs HandLandmarker: extracts 21 landmarks × 2 = 42 per hand
          (left and right hands assigned by handedness label)
        - Total: 66 + 42 + 42 = 150 features per frame

    If fewer than max_frames frames are found, zero-pads to max_frames.

    Args:
        video_path (str)        : Path to the video file (.mp4, .avi, etc.).
        pose_det               : MediaPipe PoseLandmarker instance.
        hand_det               : MediaPipe HandLandmarker instance.
        max_frames (int)        : Maximum number of frames to extract.

    Returns:
        np.ndarray: Shape (max_frames, 150) — float32 keypoint array.
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    keypoints = []   # list of frame feature vectors

    while cap.isOpened() and len(keypoints) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break   # end of video

        # Convert BGR (OpenCV default) to RGB (MediaPipe requires RGB)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_kp = []

        # ── Pose: extract 33 joints × 2 (x, y) = 66 features ─────────────────
        pose_res = pose_det.detect(img)
        if pose_res.pose_landmarks:
            # Landmarks are normalized to [0, 1] image coordinates
            for lm in pose_res.pose_landmarks[0]:
                frame_kp.extend([lm.x, lm.y])   # only x, y (ignore z/visibility)
        else:
            # No person detected → zero-fill this frame's pose features
            frame_kp.extend([0.0] * 66)

        # ── Hands: extract 21 joints × 2 per hand = 84 features total ─────────
        left_kp  = [0.0] * 42   # placeholder if left hand not detected
        right_kp = [0.0] * 42   # placeholder if right hand not detected
        hand_res = hand_det.detect(img)
        if hand_res.hand_landmarks:
            # Assign each detected hand to left or right by handedness label
            for i, hand_lms in enumerate(hand_res.hand_landmarks):
                if i >= len(hand_res.handedness):
                    break   # safety: skip if handedness count doesn't match
                side = hand_res.handedness[i][0].display_name   # "Left" or "Right"
                # Flatten the 21 landmark (x, y) pairs to a list of 42 floats
                kp   = [v for lm in hand_lms for v in [lm.x, lm.y]]
                if side == "Left":
                    left_kp = kp
                else:
                    right_kp = kp

        # Append hands in left → right order (matches the joint layout)
        frame_kp.extend(left_kp)
        frame_kp.extend(right_kp)
        keypoints.append(frame_kp)   # one entry per frame: 150 floats

    cap.release()

    # Zero-pad shorter videos to ensure uniform length
    while len(keypoints) < max_frames:
        keypoints.append([0.0] * MP_FEATURE_SIZE)

    return np.array(keypoints, dtype=np.float32)   # shape: (max_frames, 150)


# ══════════════════════════════════════════════════════════════════════════════
# YOLO BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def download_yolo_models():
    """Download YOLO body and hand pose models if not present.

    Body model: YOLOv8n-pose (standard body pose, 17 keypoints)
    Hand model: YOLO11n-pose fine-tuned for hand keypoints (21 per hand)

    Both models are saved to data/yolo_models/ after the first download.
    """
    from ultralytics import YOLO
    os.makedirs(YOLO_MODELS_DIR, exist_ok=True)

    # Download body model (YOLOv8n-pose)
    if not os.path.exists(BODY_MODEL_PATH):
        print("  Downloading YOLO body model...")
        YOLO("yolov8n-pose.pt")   # ultralytics auto-downloads to current dir
        if os.path.exists("yolov8n-pose.pt"):
            shutil.move("yolov8n-pose.pt", BODY_MODEL_PATH)   # move to our cache
        print(f"  ✅ YOLO body model saved")
    else:
        print(f"  ✅ YOLO body model already exists")

    # Download hand model (may fail if URL changes; hands will be zeros if missing)
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
    """Load YOLO body and hand pose models from disk.

    Returns:
        Tuple[YOLO, YOLO or None]: (body_model, hand_model).
            hand_model is None if the hand model file does not exist.
    """
    from ultralytics import YOLO
    body = YOLO(BODY_MODEL_PATH)
    # Only load hand model if the file is present (graceful degradation)
    hand = YOLO(YOLO_HAND_PATH) if os.path.exists(YOLO_HAND_PATH) else None
    return body, hand


def extract_yolo(video_path, body_model, hand_model, device='cpu', max_frames=MAX_FRAMES):
    """Extract per-frame skeleton keypoints from a video using YOLO pose models.

    For each frame:
        - Runs YOLOv8n-pose body model: 17 keypoints × 2 = 34 features
        - Runs YOLO11n-pose hand model: 21 keypoints × 2 per hand = 84 features
          (assigns hands by horizontal position: leftmost = right hand in mirrored view)
        - Total: 34 + 84 = 118 features per frame

    Args:
        video_path  (str)      : Path to video file.
        body_model             : Loaded YOLO body pose model.
        hand_model             : Loaded YOLO hand pose model (or None).
        device      (str)      : 'cuda' or 'cpu'.
        max_frames  (int)      : Max frames to extract.

    Returns:
        np.ndarray: Shape (max_frames, 118) — float32 keypoint array.
    """
    cap = cv2.VideoCapture(video_path)
    keypoints = []   # list of per-frame feature vectors

    while cap.isOpened() and len(keypoints) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break   # end of video

        frame_kp = []

        # ── Body: 17 keypoints × 2 = 34 features ─────────────────────────────
        body_res = body_model(frame, verbose=False, device=device)
        body_kp  = [0.0] * 34   # default zeros if no person detected
        if body_res and body_res[0].keypoints is not None:
            kps = body_res[0].keypoints.xy   # shape: (num_persons, 17, 2)
            if len(kps) > 0 and kps[0].shape[0] == 17:
                # Use only the first detected person's keypoints
                body_kp = kps[0].cpu().numpy().flatten().tolist()
        frame_kp.extend(body_kp)

        # ── Hands: 21 keypoints × 2 per hand = 84 features ───────────────────
        left_kp  = [0.0] * 42   # placeholder if left hand not detected
        right_kp = [0.0] * 42   # placeholder if right hand not detected
        if hand_model is not None:
            hand_res = hand_model(frame, verbose=False, device=device)
            if hand_res and hand_res[0].keypoints is not None:
                detected = []   # list of (x_wrist, flat_keypoints)
                for i in range(len(hand_res[0].keypoints)):
                    kps = hand_res[0].keypoints.xy[i].cpu().numpy()  # (21, 2)
                    if kps.shape[0] == 21:
                        # Use the x-coordinate of the wrist (landmark 0) to
                        # sort hands by horizontal position
                        detected.append((kps[0][0], kps.flatten().tolist()))
                # Sort left-to-right by wrist x-position
                # In a front-facing camera, the signer's right hand appears
                # on the LEFT side of the image (lower x value)
                detected.sort(key=lambda h: h[0])
                if len(detected) >= 1:
                    right_kp = detected[0][1]   # leftmost in image = signer's right
                if len(detected) >= 2:
                    left_kp  = detected[1][1]   # rightmost in image = signer's left

        # Append hands (right first, then left — matches YOLO layout convention)
        frame_kp.extend(right_kp)
        frame_kp.extend(left_kp)
        keypoints.append(frame_kp)

    cap.release()

    # Pad to max_frames with zeros if video is shorter
    while len(keypoints) < max_frames:
        keypoints.append([0.0] * YOLO_FEATURE_SIZE)

    return np.array(keypoints, dtype=np.float32)   # shape: (max_frames, 118)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(data_dir, save_dir, backend='mediapipe', device='cpu',
                  max_frames=MAX_FRAMES, subset=None):
    """Extract keypoints from all class folders and save as NumPy arrays.

    Expects the dataset folder structure:
        data_dir/
            class_name_1/
                video1.mp4
                video2.mp4
            class_name_2/
                ...

    Each class folder becomes one category. The label_map.json records the
    mapping from class name to integer index for later use during training.

    Args:
        data_dir   (str)      : Root directory containing class subfolders.
        save_dir   (str)      : Where to save X_raw.npy, y.npy, label_map.json.
        backend    (str)      : 'mediapipe' or 'yolo'.
        device     (str)      : 'cuda' or 'cpu'.
        max_frames (int)      : Max frames per video.
        subset     (int|None) : If set, only process the first N class folders
                                (useful for quick testing).

    Returns:
        Tuple[np.ndarray, np.ndarray, dict]:
            X (N, max_frames, feature_size), y (N,), label_map dict.
            Returns (None, None, None) if no classes were found.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Choose feature size based on the chosen backend
    feature_size = MP_FEATURE_SIZE if backend == 'mediapipe' else YOLO_FEATURE_SIZE

    # ── Setup backend: download models and create extractor function ──────────
    print(f"\nSetting up {backend.upper()} backend...")
    if backend == 'mediapipe':
        download_mp_models()
        pose_det, hand_det = make_mp_detectors()
        # Lambda to hide device arg (MediaPipe doesn't need it)
        extractfn = lambda vp: extract_mp(vp, pose_det, hand_det, max_frames)
    else:
        download_yolo_models()
        body_model, hand_model = make_yolo_models()
        # Lambda captures device and max_frames for YOLO
        extractfn = lambda vp: extract_yolo(vp, body_model, hand_model, device, max_frames)

    # ── Discover class folders ────────────────────────────────────────────────
    # Exclude internal folders (model caches, raw video dumps, etc.)
    EXCLUDE = {'videos', 'mp_models', 'yolo_models'}
    class_folders = sorted([
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
        and f.lower() not in EXCLUDE
    ])

    # Optional: limit to first N classes for quick testing
    if subset:
        class_folders = class_folders[:subset]

    if len(class_folders) == 0:
        print(f"❌ No class folders found. Run organize_dataset.py first.")
        return None, None, None

    # ── Print extraction configuration summary ────────────────────────────────
    print(f"\n  Backend  : {backend.upper()}")
    print(f"  Device   : {device.upper()}")
    print(f"  Features : {feature_size} per frame")
    print(f"  Classes  : {len(class_folders)}")
    print(f"  Frames   : {max_frames} per video\n")

    label_map = {}   # maps class_name → class_index
    data      = []   # accumulates one numpy array per video
    labels    = []   # accumulates integer labels for each video

    # ── Iterate over all class folders ────────────────────────────────────────
    for idx, label in enumerate(class_folders):
        folder_path = os.path.join(data_dir, label)
        label_map[label] = idx   # assign sequential integer index

        # Find all video files in this class folder
        video_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]

        print(f"[{idx+1}/{len(class_folders)}] '{label}' — {len(video_files)} videos")

        # Extract skeleton from each video with a per-class progress bar
        for vid_file in tqdm(video_files, leave=False):
            vid_path = os.path.join(folder_path, vid_file)
            try:
                skeleton = extractfn(vid_path)   # → (max_frames, feature_size)
                data.append(skeleton)
                labels.append(idx)
            except Exception as e:
                # Log and skip bad videos rather than crashing the whole run
                print(f"  [ERROR] Skipped {vid_file}: {e}")

    # ── Release MediaPipe resources (YOLO has no explicit cleanup) ─────────────
    if backend == 'mediapipe':
        pose_det.close()
        hand_det.close()

    # ── Convert lists to numpy arrays ─────────────────────────────────────────
    X = np.array(data,   dtype=np.float32)   # shape: (N, max_frames, feature_size)
    y = np.array(labels, dtype=np.int64)     # shape: (N,)

    # ── Save outputs ──────────────────────────────────────────────────────────
    np.save(os.path.join(save_dir, "X_raw.npy"), X)      # raw keypoints
    np.save(os.path.join(save_dir, "y.npy"),     y)      # class labels

    # Save label map as JSON for later use during evaluation and inference
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
# CLI ENTRY POINT
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

    # Resolve device and backend together (handles MediaPipe's GPU limitation)
    device, backend = resolve_device(args.device, args.backend)

    # Dataset root and output directory are relative to the project root
    DATA_DIR = os.path.join(ROOT, "dataset")
    SAVE_DIR = os.path.join(ROOT, "output")

    build_dataset(
        DATA_DIR, SAVE_DIR,
        backend    = backend,
        device     = device,
        max_frames = args.frames,
        subset     = args.subset
    )