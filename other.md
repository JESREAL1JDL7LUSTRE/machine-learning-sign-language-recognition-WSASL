Priority 1 — Multi-stream input (biggest gain)
Instead of just joint positions, add:

Bone vectors = joint B - joint A for each edge
Motion = joint(t) - joint(t-1) for each frame

This gives 3 streams × 102 features — SOTA models use exactly this.
Priority 2 — Better augmentation
Your current augmentation is basic. Add:

Random rotation (±15°)
Random scaling (0.8–1.2×)
Temporal cropping
Joint smoothing over time

Priority 3 — K-Fold Cross Validation
With only 201 samples, a single 70/15/15 split is unreliable. K=5 fold gives much more stable evaluation.
Priority 4 — Joint smoothing
Smooth keypoints over time before training to reduce MediaPipe detection noise.