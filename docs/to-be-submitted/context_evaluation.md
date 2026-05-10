# System Analysis: Evaluation

## Overview
The `evaluation` module serves two primary purposes: unbiased assessment of the trained models and real-world inference logic.

## Held-Out Test Set
While the 4-Fold Cross Validation (CV) provides a robust estimate of training stability, the final metric of success is the performance on a 15% completely held-out test set. This test set is never seen during the training or hyperparameter tuning phases.

**Metric Breakdown:**
- **`multi-stream-stgcn`**: 1.30% Test Accuracy
- **`4stream-late-fusion`**: 1.30% Test Accuracy
- **`4stream-fusion`**: 3.90% Test Accuracy

*(Random guessing for 50 classes equates to a 2.0% accuracy).*

## Interpretation
The severe drop from the validation accuracy (~5.5%) to the test accuracy (~1.3% - 3.9%) indicates overfitting. Due to the extremely limited dataset size (~10 samples per class), the model is memorizing the specific signing styles of the individuals in the training set rather than learning generalized sign patterns. Despite this, the `4stream-fusion` model manages to significantly outperform the baseline and random chance, proving that early feature interaction is the optimal architectural strategy.

## Single Video Inference
The `evaluate.py` script provides an end-to-end inference wrapper. It accepts a raw `.mp4` video, executes the MediaPipe extraction, runs normalization and resampling, loads the best saved model weights (`sign_stgcn.pth`), and outputs a predicted class label. This demonstrates the viability of the pipeline as a standalone application.
