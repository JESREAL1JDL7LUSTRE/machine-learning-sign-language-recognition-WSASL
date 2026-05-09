"""
dataset.py — PyTorch Dataset and Data Augmentation for Sign Language Recognition
=================================================================================

This module defines the PyTorch Dataset class and all augmentation transforms
used during training of the sign language recognition models.

Data augmentation rationale:
    Sign language datasets are typically small. Augmentation artificially
    increases effective dataset size and teaches the model to be robust to
    natural variation in:
      - Sensor jitter (add_noise)
      - Left/right-handed signing variants (horizontal_flip)
      - Incomplete or dropped video frames (random_frame_drop)
      - Natural variation in signing speed (time_warp)

All augmentations operate on the raw numpy array (T, F) BEFORE conversion
to a PyTorch tensor, and are applied ONLY during training (augment=True).

Components:
    add_noise          — Gaussian keypoint jitter
    horizontal_flip    — Mirror all x-coordinates
    temporal_resample  — Re-index to a target number of frames (used by frame_drop)
    random_frame_drop  — Remove random frames then stretch back to original length
    time_warp          — Smooth random speed variation
    SignDataset        — PyTorch Dataset that wraps a numpy array with augmentation
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Augmentation functions ────────────────────────────────────────────────────

def add_noise(skeleton, noise_level=0.01):
    """Add small Gaussian noise to all keypoints.

    Simulates sensor/detection jitter that occurs in real-world pose estimation.
    Keeps the noise level very small (0.01 in normalized coordinate space)
    so that the overall sign shape is preserved.

    Args:
        skeleton    (np.ndarray): Shape (T, F) — skeleton feature sequence.
        noise_level (float)     : Standard deviation of the added Gaussian noise.
                                  Recommended: 0.01–0.02 for normalized coordinates.

    Returns:
        np.ndarray: Shape (T, F) — noisy skeleton.
    """
    # Generate i.i.d. Gaussian noise matching the skeleton shape
    noise = np.random.randn(*skeleton.shape).astype(np.float32) * noise_level
    return skeleton + noise


def horizontal_flip(skeleton):
    """Mirror the skeleton sequence left-right by negating all x-coordinates.

    Many ASL signs are asymmetric (performed with the dominant hand), but
    some signers use the opposite hand. Flipping helps the model learn
    both variants without needing separate training examples.

    After flipping, left and right hand roles are swapped, but since the
    model treats each joint independently (via the graph), this is a valid
    data augmentation that doubles effective dataset size.

    Args:
        skeleton (np.ndarray): Shape (T, F).
                               Features are interleaved as [x0, y0, x1, y1, ...].

    Returns:
        np.ndarray: Shape (T, F) — horizontally flipped skeleton.
    """
    flipped = skeleton.copy()
    # Negate every x-coordinate (even-indexed features: 0, 2, 4, ...)
    # y-coordinates (odd indices) are unchanged — vertical position is preserved
    flipped[:, 0::2] = -flipped[:, 0::2]
    return flipped


def temporal_resample(skeleton, target_len):
    """Re-index skeleton to exactly target_len frames using nearest-neighbour sampling.

    This is a simple integer-index variant used inside random_frame_drop
    to restore the original sequence length after some frames are dropped.
    (The main preprocessing resample uses linear interpolation; this is faster
    for augmentation where exact accuracy is less critical.)

    Args:
        skeleton   (np.ndarray): Shape (T, F) — input sequence.
        target_len (int)       : Output number of frames.

    Returns:
        np.ndarray: Shape (target_len, F) — resampled sequence.
    """
    T = len(skeleton)
    if T == 0:
        return np.zeros((target_len, skeleton.shape[1]), dtype=np.float32)
    # Create target_len evenly spaced indices into the source sequence
    indices = np.linspace(0, T - 1, target_len).astype(int)
    return skeleton[indices]


def random_frame_drop(skeleton, drop_prob=0.1):
    """Randomly remove frames then resample back to the original length.

    Teaches the model to handle missing or corrupted frames gracefully.
    This is common in real-world video: dropped frames, network issues,
    or occlusion can cause detection failures.

    Process:
        1. For each frame, independently keep it with probability (1 - drop_prob).
        2. If fewer than 2 frames survive, return original (safety guard).
        3. Resample the kept frames back to the original T length.

    Args:
        skeleton  (np.ndarray): Shape (T, F).
        drop_prob (float)     : Probability of dropping each frame (0.0–1.0).
                                Recommended: 0.1–0.2.

    Returns:
        np.ndarray: Shape (T, F) — sequence with some frames randomly removed
                    then resampled back to T.
    """
    T = len(skeleton)
    # Bernoulli mask: True = keep, False = drop
    mask = np.random.rand(T) > drop_prob
    kept = skeleton[mask]   # keep only the surviving frames

    # Safety: don't drop almost everything (need ≥2 frames to resample)
    if len(kept) < 2:
        return skeleton   # return original unchanged

    return temporal_resample(kept, target_len=T)


def time_warp(skeleton, sigma=0.2):
    """Apply slight speed variation by randomly warping the time axis.

    Simulates different signing speeds: some signers sign faster or slower,
    or accelerate/decelerate during a sign. Time warping helps the model
    generalize across these natural speed variations.

    Process:
        1. Generate a smooth random warp curve by cumsum of a Gaussian walk.
        2. Normalize the warp curve to [0, 1].
        3. Use it to sample frames from the original sequence at warped positions.

    Args:
        skeleton (np.ndarray): Shape (T, F).
        sigma    (float)     : Controls warp strength. Larger σ = more distortion.
                               Recommended range: 0.1–0.3.

    Returns:
        np.ndarray: Shape (T, F) — time-warped sequence (same number of frames,
                    but sampled at non-uniform intervals from the original).
    """
    T = len(skeleton)
    # Create a smooth random warp path by cumulative sum of Gaussian noise
    warp = np.cumsum(np.random.randn(T) * sigma)
    # Normalize to range [0, 1]
    warp -= warp.min()
    warp /= (warp.max() + 1e-6)   # avoid division by zero if all values are the same
    # Map normalized warp to integer frame indices in [0, T-1]
    warped_indices = (warp * (T - 1)).astype(int)
    warped_indices = np.clip(warped_indices, 0, T - 1)
    return skeleton[warped_indices]


# ── Dataset ───────────────────────────────────────────────────────────────────

class SignDataset(Dataset):
    """PyTorch Dataset for Isolated Sign Language Recognition.

    Wraps numpy arrays (X, y) and optionally applies random augmentation
    to each sample during loading. This is used by the LSTM/TCN model
    (train.py equivalent for the flat-feature models).

    The four-stream ST-GCN training (train.py, main.py) uses its own
    FourStreamDataset with a different (heavier) augmentation pipeline.

    Args:
        X       (np.ndarray): Shape (N, T, F) — skeleton sequence dataset.
                              F = 150 for MediaPipe (raw) or 102 for filtered.
        y       (np.ndarray): Shape (N,) — integer class labels.
        augment (bool)      : If True, applies random augmentations.
                              Use True for training, False for val/test.

    __getitem__ returns:
        Tuple[Tensor, Tensor]:
            - x: shape (T, F), float32 — optionally augmented skeleton
            - y: scalar, long — class label
    """

    def __init__(self, X, y, augment=False):
        # Store as float32 / int64 (required by PyTorch for inputs and targets)
        self.X       = X.astype(np.float32)
        self.y       = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Load one sample, optionally augmenting it.

        Each augmentation is applied independently at 30–50% probability,
        meaning multiple augmentations can be combined in a single sample.

        Augmentation probabilities:
            50% — Gaussian noise (simulates detection jitter)
            50% — Horizontal flip (simulates opposite-handed signing)
            30% — Random frame drop (simulates missing/dropped frames)
            30% — Time warp (simulates speed variation)

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[Tensor, Tensor]: (x, label) — float32 tensor and int64 label.
        """
        # Make a copy so augmentation doesn't modify the original stored array
        x = self.X[idx].copy()   # shape: (T, F)

        if self.augment:
            # Each augmentation is applied independently (not mutually exclusive)
            if np.random.rand() < 0.5:
                x = add_noise(x)             # keypoint jitter

            if np.random.rand() < 0.5:
                x = horizontal_flip(x)       # mirror left-right

            if np.random.rand() < 0.3:
                x = random_frame_drop(x, drop_prob=0.1)  # drop ~10% of frames

            if np.random.rand() < 0.3:
                x = time_warp(x, sigma=0.15)  # speed variation

        return (
            torch.tensor(x,           dtype=torch.float32),  # (T, F)
            torch.tensor(self.y[idx], dtype=torch.long)       # scalar label
        )