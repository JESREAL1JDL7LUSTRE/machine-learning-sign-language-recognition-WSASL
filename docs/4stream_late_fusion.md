# 4-Stream Late Fusion ST-GCN

## Overview
The **4-Stream Late Fusion** model is an experimental variant of the ST-GCN architecture designed to evaluate whether adding the "Bone Motion" (angular velocity) stream improves accuracy without requiring the complexity of a shared early-fusion classifier.

## Architecture
The architecture consists of four entirely independent ST-GCN pipelines, each processing a distinct data stream:
1. **Joint Stream**: Processes absolute `(x, y)` skeletal coordinates.
2. **Bone Stream**: Processes relative limb orientations (`target_joint - source_joint`).
3. **Motion Stream**: Processes frame-to-frame joint velocity.
4. **Bone Motion Stream**: Processes frame-to-frame angular limb velocity.

### Late Fusion Logic
"Late Fusion" refers to the point at which the network aggregates the data from these four streams.
- Each of the four backbones independently outputs a probability distribution (logits) across the 50 possible sign classes.
- The final prediction is made by simply summing the four sets of logits together:
  `Final Logits = Logits(Joint) + Logits(Bone) + Logits(Motion) + Logits(Bone Motion)`

## Code Location
This model is instantiated in `main.py` by calling `FourStreamSTGCN` from `models/st_gcn_twostream.py` with the parameter `early_fusion=False`.

## Results
- **Cross-Validation Mean**: 5.57%
- **Test Accuracy**: 1.30%

## Conclusion
The late fusion variant performed exceptionally poorly on the test set compared to the early fusion variant. This indicates that while the bone motion stream contains valuable information, treating it as an independent voter (late fusion) is insufficient. The network requires a shared classifier (early fusion) to correlate the angular velocity with the absolute positions and orientations of the other streams.
