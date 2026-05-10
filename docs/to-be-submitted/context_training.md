# System Analysis: Training

## Overview
The `training` module manages the optimization of the ST-GCN models. Given the extremely small dataset (~524 videos across 50 classes), the training strategy incorporates significant regularization to prevent overfitting.

## Core Setup
- **Optimizer**: AdamW optimizer with weight decay (0.05).
- **Learning Rate Schedule**: Warmup for the first 10 epochs followed by a Cosine Annealing decay schedule.
- **Loss Function**: Label-smoothed Cross Entropy (ε=0.1) paired with inverse-frequency class weighting to penalize overconfidence and handle imbalanced class distributions.

## Regularization Strategies
- **Data Augmentation**: Robust transformations are applied dynamically during training, including temporal cropping, temporal dropout (masking 15% of frames), random scaling, rotation, and Gaussian noise injection.
- **DropGraph / Dropout**: Spatial dropout techniques are used inside the ST-GCN blocks.
- **Stochastic Weight Averaging (SWA)**: After epoch 50, model weights are averaged across multiple epochs to find a wider, flatter local minimum, improving generalization on unseen validation data.

## K-Fold Cross Validation
The system utilizes 4-Fold Stratified Cross-Validation on 85% of the data. 
- Normalization (mean subtraction and standard deviation scaling) is computed *only* on the training split of each fold and applied to the validation set to prevent data leakage.
- The training loop tracks validation accuracy and incorporates an early stopping mechanism (patience = 35) to halt training when improvements plateau.
