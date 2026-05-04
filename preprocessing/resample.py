import os

import numpy as np


def valid_length(skeleton, eps=1e-6):
    """Return effective sequence length excluding trailing all-zero frames."""
    frame_energy = np.sum(np.abs(skeleton), axis=1)
    valid = np.where(frame_energy > eps)[0]
    if len(valid) == 0:
        return 0
    return int(valid[-1]) + 1


def temporal_resample(skeleton, target_len=64):
    """
    Resample a sequence to target_len using linear interpolation over valid frames.
    """
    T, F = skeleton.shape
    if target_len <= 0:
        raise ValueError("target_len must be > 0")

    T_valid = valid_length(skeleton)
    if T_valid == 0:
        return np.zeros((target_len, F), dtype=np.float32)

    seq = skeleton[:T_valid].astype(np.float32)
    if T_valid == target_len:
        return seq

    src_t = np.arange(T_valid, dtype=np.float32)
    dst_t = np.linspace(0, T_valid - 1, target_len, dtype=np.float32)

    out = np.zeros((target_len, F), dtype=np.float32)
    for f in range(F):
        out[:, f] = np.interp(dst_t, src_t, seq[:, f])
    return out


def resample_dataset(X, target_len=64):
    return np.array([temporal_resample(X[i], target_len=target_len) for i in range(len(X))], dtype=np.float32)


def _resample_if_exists(output_dir, in_name, out_name, target_len):
    in_path = os.path.join(output_dir, in_name)
    if not os.path.exists(in_path):
        return False

    X = np.load(in_path)
    X_resampled = resample_dataset(X, target_len=target_len)
    np.save(os.path.join(output_dir, out_name), X_resampled)
    print(f"Saved {out_name}: {X_resampled.shape}")
    return True


if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    OUTPUT_DIR = os.path.join(ROOT, "output")
    TARGET_LEN = 64

    print(f"Resampling all available streams to {TARGET_LEN} frames ...")

    # Primary joint stream used by training loader fallback order.
    has_joint = _resample_if_exists(OUTPUT_DIR, "X_normalized.npy", "X_final.npy", TARGET_LEN)
    if not has_joint:
        has_joint = _resample_if_exists(OUTPUT_DIR, "X_raw.npy", "X_final.npy", TARGET_LEN)

    _resample_if_exists(OUTPUT_DIR, "X_bones.npy", "X_bones.npy", TARGET_LEN)
    _resample_if_exists(OUTPUT_DIR, "X_motion.npy", "X_motion.npy", TARGET_LEN)
    _resample_if_exists(OUTPUT_DIR, "X_bone_motion.npy", "X_bone_motion.npy", TARGET_LEN)

    if not has_joint:
        print("No joint stream found. Run extract.py and normalize.py first.")
        raise SystemExit(1)

    print("Resampling complete.")
