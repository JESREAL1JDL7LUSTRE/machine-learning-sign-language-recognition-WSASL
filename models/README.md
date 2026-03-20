# Models Module

## Purpose
Defines machine learning models used for sign language recognition.

## Responsibilities
- Implement model architectures
- Define forward pass
- Keep models modular and reusable

## Examples
- MLP (baseline)
- CNN (image-based)
- GCN (graph-based)
- DGAT (advanced, optional)

## Input
- Feature vectors (flattened keypoints or images)

## Output
- Class probabilities (Softmax)

## Notes
- Start with a SIMPLE model first (MLP baseline)
- Only move to complex models (GCN/DGAT) after baseline works
- Keep model independent from training logic

## Future Extensions
- Attention mechanisms
- Multi-stream models