1. Massive Data Augmentation (Do this first)
Since you can't easily record thousands of new videos, you must "fake" them. Before feeding your .npy data into the LSTM, apply random transformations to the keypoints in your training script:
Scaling: Multiply coordinates by a random factor between 0.9 and 1.1 (simulates standing closer/further).
Rotation: Rotate the keypoints slightly (simulates camera tilt).
Noise: Add tiny amounts of Gaussian noise to each keypoint.
Time Stretching: Randomly skip or duplicate a few frames (simulates faster/slower signing).
2. Simplify the Input (Feature Engineering)
Your input shape is (64, 150). 150 features per frame is a lot for a tiny dataset to handle.
Relative Coordinates: Don't use raw X, Y. Subtract the wrist coordinate from all finger coordinates. This makes the data "location independent."
Drop unnecessary points: Are you using face mesh? For Sign Language, you usually only need the Pose (upper body) and Both Hands. If you are including 400+ face landmarks, you are drowning the "signal" in "noise."
