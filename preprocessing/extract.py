import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

mp_holistic = mp.solutions.holistic

# Total features per frame:
# Pose: 33 landmarks * 2 = 66
# Left Hand: 21 landmarks * 2 = 42
# Right Hand: 21 landmarks * 2 = 42
# TOTAL = 150

FEATURE_SIZE = 150
MAX_FRAMES = 64


def extract_skeleton(video_path, max_frames=MAX_FRAMES):
    """
    Extract pose + hand keypoints from a video using MediaPipe Holistic.
    Returns: numpy array of shape (T, 150)
    """
    cap = cv2.VideoCapture(video_path)
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False
    )

    keypoints = []

    while cap.isOpened() and len(keypoints) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        frame_kp = []

        # --- Pose (33 landmarks x 2 = 66) ---
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_kp.extend([lm.x, lm.y])
        else:
            frame_kp.extend([0.0] * 66)

        # --- Left Hand (21 landmarks x 2 = 42) ---
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_kp.extend([lm.x, lm.y])
        else:
            frame_kp.extend([0.0] * 42)

        # --- Right Hand (21 landmarks x 2 = 42) ---
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_kp.extend([lm.x, lm.y])
        else:
            frame_kp.extend([0.0] * 42)

        keypoints.append(frame_kp)

    cap.release()
    holistic.close()

    # Pad if video is shorter than max_frames
    while len(keypoints) < max_frames:
        keypoints.append([0.0] * FEATURE_SIZE)

    return np.array(keypoints, dtype=np.float32)  # shape: (T, 150)


def build_dataset(data_dir, save_dir, max_frames=MAX_FRAMES):
    """
    Walk through data_dir (class folders with videos),
    extract skeletons, and save X.npy + y.npy to save_dir.

    Expected structure:
        data_dir/
            hello/
                vid1.mp4
                vid2.mp4
            thanks/
                vid3.mp4
    """
    os.makedirs(save_dir, exist_ok=True)

    label_map = {}
    data = []
    labels = []

    class_folders = sorted(os.listdir(data_dir))

    for idx, label in enumerate(class_folders):
        folder_path = os.path.join(data_dir, label)

        if not os.path.isdir(folder_path):
            continue

        label_map[label] = idx
        video_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]

        print(f"\n[{idx}] Processing class: '{label}' ({len(video_files)} videos)")

        for vid_file in tqdm(video_files):
            vid_path = os.path.join(folder_path, vid_file)

            try:
                skeleton = extract_skeleton(vid_path, max_frames=max_frames)
                data.append(skeleton)
                labels.append(idx)
            except Exception as e:
                print(f"  [ERROR] Skipped {vid_file}: {e}")

    X = np.array(data, dtype=np.float32)   # (N, T, 150)
    y = np.array(labels, dtype=np.int64)   # (N,)

    np.save(os.path.join(save_dir, "X_raw.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"), y)

    # Save label map for reference
    import json
    with open(os.path.join(save_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n✅ Dataset saved to '{save_dir}'")
    print(f"   X shape : {X.shape}")
    print(f"   y shape : {y.shape}")
    print(f"   Classes : {label_map}")

    return X, y, label_map


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR = "../dataset"   # raw videos folder
    SAVE_DIR = "../output"    # where X_raw.npy and y.npy will be saved

    build_dataset(DATA_DIR, SAVE_DIR)