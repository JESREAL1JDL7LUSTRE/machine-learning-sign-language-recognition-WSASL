# Evaluation Module

## Purpose
Evaluates model performance on the test dataset.

## Responsibilities
- Compute metrics:
  - Accuracy (Top-1)
  - Top-5 Accuracy (optional)
  - Precision, Recall, F1-score
- Generate confusion matrix

## Input
- Trained model
- Test dataset

## Output
- Metrics (printed and/or saved)
- Visualization (optional)

## Notes
- This module should NOT modify model weights
- Ensure evaluation uses unseen test data only
- Keep evaluation reproducible

## Future Extensions
- Detailed error analysis
- Per-class performance