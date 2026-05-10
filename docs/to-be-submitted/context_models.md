# System Analysis: Models

## Overview
The `models` module defines the Spatial-Temporal Graph Convolutional Network (ST-GCN) architectures. Rather than treating video as a 3D pixel volume, ST-GCN treats the human skeleton as a graph. Spatial convolutions aggregate data between physically connected joints (e.g., wrist to elbow), while temporal convolutions track these joints over time.

## Compared Architectures

### 1. Multi-Stream ST-GCN (3-Stream Late Fusion)
- **Inputs**: Joint positions, Bone vectors, Joint motion.
- **Fusion**: Late Fusion. Each of the three inputs is passed through its own independent 6-block ST-GCN backbone. The three resulting output logits (class probabilities) are summed together.
- **Role**: Baseline model representing standard multi-stream setups.

### 2. Four-Stream Late Fusion ST-GCN
- **Inputs**: Joint, Bone, Motion, and *Bone Motion* (angular velocity).
- **Fusion**: Late Fusion. Similar to the 3-stream model, but with a fourth independent backbone.
- **Role**: Tests whether the addition of angular velocity data strictly improves performance without changing the underlying integration logic.

### 3. Four-Stream Early Fusion ST-GCN (Best Model)
- **Inputs**: Joint, Bone, Motion, Bone Motion.
- **Fusion**: Early Fusion. The inputs pass through independent ST-GCN feature extractors, but instead of outputting class probabilities, they output 128-dimensional feature vectors. These vectors are concatenated (forming a 512-dimensional vector) and passed through a shared Multi-Layer Perceptron (MLP) classifier.
- **Role**: This model achieved the highest test accuracy (3.90%). Early fusion allows the dense classifier to cross-reference the features (e.g., matching arm orientation with angular velocity) before making a decision, proving superior for complex sign language gestures.
