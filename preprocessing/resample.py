"""
resample.py — Temporal Resampling to Fixed Sequence Length
===========================================================

This is the third and final step of the preprocessing pipeline.
It ensures every skeleton sequence has exactly target_len frames,
regardless of the original video duration.

Why resampling is needed:
    Videos can have different frame counts depending on their duration and
    frame rate. Neural networks require fixed-length inputs. Resampling
    stretches or compresses each sequence to a uniform length using linear
    interpolation, which is smoother than repeating or truncating frames.

Approach:
    1. Find the 'effective' length: the last non-zero frame index.
    2. Interpolate (stretch/compress) only the valid portion of the sequence.
    3. Output exactly target_len frames.

This module applies resampling to all four data streams produced by normalize.py:
    - X_normalized.npy → X_final.npy  (joint positions, primary training stream)
    - X_bones.npy      → X_bones.npy  (in-place)
    - X_motion.npy     → X_motion.npy (in-place)
    - X_bone_motion.npy→ X_bone_motion.npy (in-place)

Usage:
    python preprocessing/resample.py

Functions:
    valid_length      — Find effective (non-padded) length of a sequence.
    temporal_resample — Resample a single sequence to target_len frames.
    resample_dataset  — Apply temporal_resample to an entire dataset array.
"""

import os

import numpy as np


def valid_length(skeleton, eps=1e-6):
    """Find the effective sequence length, excluding trailing all-zero frames.

    Videos shorter than max_frames were zero-padded in extract.py.
    This function finds the last frame index with any non-zero signal,
    so resampling operates only on the actual video content.

    Args:
        skeleton (np.ndarray): Shape (T, F) — one video's raw features.
        eps      (float)     : Frames with total absolute value below this
                               threshold are considered zero-padded.

    Returns:
        int: Number of valid (non-padded) frames. Returns 0 if fully zero.
    """
    # Sum absolute values across all features per frame to get per-frame 'energy'
    frame_energy = np.sum(np.abs(skeleton), axis=1)   # shape: (T,)
    # Find indices of frames that have any content
    valid = np.where(frame_energy > eps)[0]
    if len(valid) == 0:
        return 0   # completely empty sequence
    # Return index of last valid frame + 1 (exclusive end index)
    return int(valid[-1]) + 1


def temporal_resample(skeleton, target_len=64):
    """Resample a skeleton sequence to exactly target_len frames.

    Steps:
        1. Find valid_length to skip trailing zeros.
        2. If valid_length == 0, return all-zero output.
        3. If valid_length == target_len, return unchanged.
        4. Otherwise, use np.interp to linearly interpolate each feature
           along the time axis from the original grid to the target grid.

    This produces smooth frame sequences for both:
        - Short videos (< target_len frames): stretched / upsampled
        - Long videos  (> target_len frames): compressed / downsampled

    Args:
        skeleton   (np.ndarray): Shape (T, F) — one video's features.
        target_len (int)       : Desired output length.

    Returns:
        np.ndarray: Shape (target_len, F) — resampled sequence, float32.

    Raises:
        ValueError: If target_len <= 0.
    """
    T, F = skeleton.shape
    if target_len <= 0:
        raise ValueError("target_len must be > 0")

    # Find the end of the actual video content (exclude zero-padding)
    T_valid = valid_length(skeleton)
    if T_valid == 0:
        # Nothing to resample — return zero-filled target
        return np.zeros((target_len, F), dtype=np.float32)

    # Use only the valid portion of the sequence
    seq = skeleton[:T_valid].astype(np.float32)
    if T_valid == target_len:
        return seq   # already the right length — no interpolation needed

    # Source time grid: evenly spaced indices over the valid frames
    src_t = np.arange(T_valid, dtype=np.float32)
    # Destination time grid: evenly spaced indices over target length,
    # scaled to match the source range
    dst_t = np.linspace(0, T_valid - 1, target_len, dtype=np.float32)

    # Interpolate each feature channel independently over time
    out = np.zeros((target_len, F), dtype=np.float32)
    for f in range(F):
        out[:, f] = np.interp(dst_t, src_t, seq[:, f])
    return out


def resample_dataset(X, target_len=64):
    """Apply temporal_resample to every sample in a dataset array.

    Args:
        X          (np.ndarray): Shape (N, T, F) — full dataset.
        target_len (int)       : Output length per sample.

    Returns:
        np.ndarray: Shape (N, target_len, F) — resampled dataset, float32.
    """
    return np.array(
        [temporal_resample(X[i], target_len=target_len) for i in range(len(X))],
        dtype=np.float32
    )


def _resample_if_exists(output_dir, in_name, out_name, target_len):
    """Load a .npy file, resample it, and save the result.

    Helper used by the CLI to process each data stream file.
    Skips silently if the input file does not exist.

    Args:
        output_dir (str): Directory containing the .npy files.
        in_name    (str): Input filename (e.g. 'X_normalized.npy').
        out_name   (str): Output filename (e.g. 'X_final.npy').
        target_len (int): Target number of frames.

    Returns:
        bool: True if the file was found and resampled, False if not found.
    """
    in_path = os.path.join(output_dir, in_name)
    if not os.path.exists(in_path):
        return False   # file doesn't exist — skip

    X = np.load(in_path)
    X_resampled = resample_dataset(X, target_len=target_len)
    np.save(os.path.join(output_dir, out_name), X_resampled)
    print(f"Saved {out_name}: {X_resampled.shape}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    OUTPUT_DIR = os.path.join(ROOT, "output")
    TARGET_LEN = 64   # standard sequence length used by all models

    print(f"Resampling all available streams to {TARGET_LEN} frames ...")

    # ── Primary joint stream ───────────────────────────────────────────────────
    # Try X_normalized first (post-normalization), fall back to X_raw if needed.
    # Output is always X_final.npy (the file train.py looks for first).
    has_joint = _resample_if_exists(OUTPUT_DIR, "X_normalized.npy", "X_final.npy", TARGET_LEN)
    if not has_joint:
        has_joint = _resample_if_exists(OUTPUT_DIR, "X_raw.npy", "X_final.npy", TARGET_LEN)

    # ── Auxiliary streams (in-place resampling) ───────────────────────────────
    # Each file is overwritten with its resampled version
    _resample_if_exists(OUTPUT_DIR, "X_bones.npy",       "X_bones.npy",       TARGET_LEN)
    _resample_if_exists(OUTPUT_DIR, "X_motion.npy",      "X_motion.npy",      TARGET_LEN)
    _resample_if_exists(OUTPUT_DIR, "X_bone_motion.npy", "X_bone_motion.npy", TARGET_LEN)

    # Error out if no joint data was found at all
    if not has_joint:
        print("No joint stream found. Run extract.py and normalize.py first.")
        raise SystemExit(1)

    print("Resampling complete.")
