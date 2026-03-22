import numpy as np
import torch
from torch.utils.data import Dataset


# ── Augmentation functions ────────────────────────────────────────────────────

def add_noise(skeleton, noise_level=0.01):
    """
    Add small Gaussian noise to all keypoints.
    Simulates sensor/detection jitter.

    Args:
        skeleton    : (T, F) numpy array
        noise_level : std of the noise (keep small, e.g. 0.01)
    """
    noise = np.random.randn(*skeleton.shape).astype(np.float32) * noise_level
    return skeleton + noise


def horizontal_flip(skeleton):
    """
    Mirror the sequence left-right by negating all x-coordinates.
    Useful because many signs have left/right variants.

    Args:
        skeleton : (T, F) numpy array
                   Features are ordered as [x0, y0, x1, y1, ...]
    """
    flipped = skeleton.copy()
    # Negate every x-coordinate (even indices: 0, 2, 4, ...)
    flipped[:, 0::2] = -flipped[:, 0::2]
    return flipped


def temporal_resample(skeleton, target_len):
    """Re-index skeleton to target_len frames (used by frame_drop)."""
    T = len(skeleton)
    if T == 0:
        return np.zeros((target_len, skeleton.shape[1]), dtype=np.float32)
    indices = np.linspace(0, T - 1, target_len).astype(int)
    return skeleton[indices]


def random_frame_drop(skeleton, drop_prob=0.1):
    """
    Randomly remove frames then resample back to the original length.
    Teaches the model to handle missing / noisy temporal data.

    Args:
        skeleton  : (T, F) numpy array
        drop_prob : probability of dropping each frame
    """
    T = len(skeleton)
    mask = np.random.rand(T) > drop_prob
    kept = skeleton[mask]

    if len(kept) < 2:
        return skeleton  # safety — don't drop almost everything

    return temporal_resample(kept, target_len=T)


def time_warp(skeleton, sigma=0.2):
    """
    Slight speed variation: randomly stretch / compress time.
    Simulates different signing speeds.

    Args:
        skeleton : (T, F) numpy array
        sigma    : controls warp strength (0.1–0.3 is reasonable)
    """
    T = len(skeleton)
    # Create a smooth random warp curve
    warp = np.cumsum(np.random.randn(T) * sigma)
    warp -= warp.min()
    warp /= (warp.max() + 1e-6)
    warped_indices = (warp * (T - 1)).astype(int)
    warped_indices = np.clip(warped_indices, 0, T - 1)
    return skeleton[warped_indices]


# ── Dataset ───────────────────────────────────────────────────────────────────

class SignDataset(Dataset):
    """
    PyTorch Dataset for Isolated Sign Language Recognition.

    Args:
        X       : numpy array (N, T, 150)
        y       : numpy array (N,)
        augment : whether to apply random augmentation (use True for training only)
    """

    def __init__(self, X, y, augment=False):
        self.X       = X.astype(np.float32)
        self.y       = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()   # (T, 150)

        if self.augment:
            # Each augmentation applied independently with 50% chance
            if np.random.rand() < 0.5:
                x = add_noise(x)

            if np.random.rand() < 0.5:
                x = horizontal_flip(x)

            if np.random.rand() < 0.3:
                x = random_frame_drop(x, drop_prob=0.1)

            if np.random.rand() < 0.3:
                x = time_warp(x, sigma=0.15)

        return (
            torch.tensor(x,           dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long)
        )