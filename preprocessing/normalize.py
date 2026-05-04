import os

import numpy as np
from scipy.ndimage import gaussian_filter1d

UPPER_BODY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24]

UPPER_BODY_FEAT = []
for j in UPPER_BODY_JOINTS:
    UPPER_BODY_FEAT.extend([j * 2, j * 2 + 1])

HAND_FEAT = list(range(66, 150))
KEEP_INDICES = UPPER_BODY_FEAT + HAND_FEAT  # 102 total -> 51 joints x 2

_F_NOSE = 0
_F_LSH = 1
_F_RSH = 2
_F_LEL = 3
_F_REL = 4
_F_LWR = 5
_F_RWR = 6
_F_LHI = 7
_F_RHI = 8

FILT_LEFT_HAND_OFFSET = 18
FILT_RIGHT_HAND_OFFSET = 60
HAND_JOINTS = 21

# (parent, child): bone vector = child - parent in filtered 51-joint space
BONE_EDGES = [
    (0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8),
    (9, 10), (10, 11), (11, 12), (12, 13),
    (9, 14), (14, 15), (15, 16), (16, 17),
    (9, 18), (18, 19), (19, 20), (20, 21),
    (9, 22), (22, 23), (23, 24), (24, 25),
    (9, 26), (26, 27), (27, 28), (28, 29),
    (30, 31), (31, 32), (32, 33), (33, 34),
    (30, 35), (35, 36), (36, 37), (37, 38),
    (30, 39), (39, 40), (40, 41), (41, 42),
    (30, 43), (43, 44), (44, 45), (45, 46),
    (30, 47), (47, 48), (48, 49), (49, 50),
]


def filter_joints(X):
    return X[:, :, KEEP_INDICES].astype(np.float32)


def _interpolate_joint_track(track_xy):
    """Interpolate one joint trajectory of shape (T, 2)."""
    track_xy = track_xy.astype(np.float32)
    T = track_xy.shape[0]

    valid = ~np.all(np.isclose(track_xy, 0.0), axis=1)
    if not np.any(valid):
        return track_xy

    valid_idx = np.where(valid)[0]
    full_idx = np.arange(T)

    out = track_xy.copy()
    out[:, 0] = np.interp(full_idx, valid_idx, track_xy[valid_idx, 0])
    out[:, 1] = np.interp(full_idx, valid_idx, track_xy[valid_idx, 1])
    return out


def clean_missing_keypoints(skeleton):
    """
    Fill missing keypoints by per-joint temporal interpolation.
    This is more robust than frame-level filling when only some joints disappear.
    """
    skeleton = skeleton.astype(np.float32).copy()
    T, F = skeleton.shape
    V = F // 2

    joints = skeleton.reshape(T, V, 2)
    for v in range(V):
        joints[:, v, :] = _interpolate_joint_track(joints[:, v, :])

    return joints.reshape(T, F)


def make_relative_hands(frame):
    frame = frame.copy()

    lx = frame[FILT_LEFT_HAND_OFFSET]
    ly = frame[FILT_LEFT_HAND_OFFSET + 1]
    if not (lx == 0.0 and ly == 0.0):
        for j in range(HAND_JOINTS):
            frame[FILT_LEFT_HAND_OFFSET + j * 2] -= lx
            frame[FILT_LEFT_HAND_OFFSET + j * 2 + 1] -= ly

    rx = frame[FILT_RIGHT_HAND_OFFSET]
    ry = frame[FILT_RIGHT_HAND_OFFSET + 1]
    if not (rx == 0.0 and ry == 0.0):
        for j in range(HAND_JOINTS):
            frame[FILT_RIGHT_HAND_OFFSET + j * 2] -= rx
            frame[FILT_RIGHT_HAND_OFFSET + j * 2 + 1] -= ry

    return frame


def _joint_xy(frame, joint_idx):
    x = frame[joint_idx * 2]
    y = frame[joint_idx * 2 + 1]
    if np.isclose(x, 0.0) and np.isclose(y, 0.0):
        return None
    return np.array([x, y], dtype=np.float32)


def _pick_center(frame):
    lsh = _joint_xy(frame, _F_LSH)
    rsh = _joint_xy(frame, _F_RSH)
    if lsh is not None and rsh is not None:
        return 0.5 * (lsh + rsh)

    lhi = _joint_xy(frame, _F_LHI)
    rhi = _joint_xy(frame, _F_RHI)
    if lhi is not None and rhi is not None:
        return 0.5 * (lhi + rhi)

    nose = _joint_xy(frame, _F_NOSE)
    if nose is not None:
        return nose

    return None


def _pick_scale(frame):
    lsh = _joint_xy(frame, _F_LSH)
    rsh = _joint_xy(frame, _F_RSH)
    if lsh is not None and rsh is not None:
        d = np.linalg.norm(lsh - rsh)
        if d > 1e-6:
            return d

    lwr = _joint_xy(frame, _F_LWR)
    lel = _joint_xy(frame, _F_LEL)
    if lwr is not None and lel is not None:
        d = np.linalg.norm(lwr - lel)
        if d > 1e-6:
            return d

    rwr = _joint_xy(frame, _F_RWR)
    rel = _joint_xy(frame, _F_REL)
    if rwr is not None and rel is not None:
        d = np.linalg.norm(rwr - rel)
        if d > 1e-6:
            return d

    return 1.0


def normalize_skeleton(skeleton):
    skeleton = skeleton.copy().astype(np.float32)
    for t in range(skeleton.shape[0]):
        frame = skeleton[t]

        center = _pick_center(frame)
        if center is None:
            continue

        frame[0::2] -= center[0]
        frame[1::2] -= center[1]

        scale = _pick_scale(frame)
        frame /= scale

        frame = make_relative_hands(frame)
        skeleton[t] = frame

    return skeleton


def normalize_dataset(X):
    return np.array([normalize_skeleton(X[i]) for i in range(len(X))], dtype=np.float32)


def smooth_skeleton(skeleton, sigma=0.8):
    return gaussian_filter1d(skeleton, sigma=sigma, axis=0).astype(np.float32)


def smooth_dataset(X, sigma=0.8):
    return np.array([smooth_skeleton(X[i], sigma) for i in range(len(X))], dtype=np.float32)


def compute_bone_vectors(X):
    """Bone = child_joint - parent_joint. Shape: (N, T, 102)"""
    N, T, F = X.shape
    V = F // 2
    joints = X.reshape(N, T, V, 2)
    bones = np.zeros_like(joints)
    for parent, child in BONE_EDGES:
        if parent < V and child < V:
            bones[:, :, child, :] = joints[:, :, child, :] - joints[:, :, parent, :]
    return bones.reshape(N, T, F).astype(np.float32)


def compute_motion(X):
    """Frame-to-frame delta. motion[t] = X[t+1] - X[t]. Shape: (N, T, F)"""
    motion = np.zeros_like(X, dtype=np.float32)
    motion[:, :-1] = X[:, 1:] - X[:, :-1]
    return motion


if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    OUTPUT_DIR = os.path.join(ROOT, "output")

    X_raw_path = os.path.join(OUTPUT_DIR, "X_raw.npy")
    if not os.path.exists(X_raw_path):
        print("X_raw.npy not found. Run extract.py first.")
        raise SystemExit(1)

    print("Loading X_raw.npy ...")
    X_raw = np.load(X_raw_path)
    print(f"   Shape: {X_raw.shape}")

    print("\nStep 1 - Cleaning missing keypoints with per-joint interpolation ...")
    X_clean = np.array([clean_missing_keypoints(X_raw[i]) for i in range(len(X_raw))], dtype=np.float32)

    print("\nStep 2 - Filtering joints (150 -> 102) ...")
    X_filt = filter_joints(X_clean)
    print(f"   Shape: {X_filt.shape}")

    print("\nStep 3 - Normalizing (center + scale + relative hands) ...")
    X_norm = normalize_dataset(X_filt)

    print("\nStep 4 - Temporal smoothing ...")
    X_smooth = smooth_dataset(X_norm, sigma=0.8)

    print("\nStep 5 - Computing bone vectors ...")
    X_bones = compute_bone_vectors(X_smooth)

    print("\nStep 6 - Computing joint motion ...")
    X_motion = compute_motion(X_smooth)

    print("\nStep 7 - Computing bone motion ...")
    X_bone_motion = compute_motion(X_bones)

    np.save(os.path.join(OUTPUT_DIR, "X_normalized.npy"), X_smooth)
    np.save(os.path.join(OUTPUT_DIR, "X_bones.npy"), X_bones)
    np.save(os.path.join(OUTPUT_DIR, "X_motion.npy"), X_motion)
    np.save(os.path.join(OUTPUT_DIR, "X_bone_motion.npy"), X_bone_motion)

    print(f"\nSaved all 4 streams to {OUTPUT_DIR}")
    print(f"   X_normalized  : {X_smooth.shape}      (joint positions)")
    print(f"   X_bones       : {X_bones.shape}       (bone vectors)")
    print(f"   X_motion      : {X_motion.shape}      (joint motion)")
    print(f"   X_bone_motion : {X_bone_motion.shape} (bone motion)")
