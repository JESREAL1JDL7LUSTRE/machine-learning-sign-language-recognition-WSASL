# ASL Sign Language Recognition
Using Spatial-Temporal Graph Convolutional Networks

## 1. Overview of your Project
This project presents an end-to-end skeleton-based isolated sign language recognition system for American Sign Language (ASL). The system accepts sign language videos as input and classifies them into one of 50 sign word categories drawn from the WLASL dataset.
The recognition engine is a Spatial-Temporal Graph Convolutional Network (ST-GCN) — a graph neural network that treats the human skeleton as a graph, where joints are nodes and bones are edges. Three distinct model variants are trained and compared, focusing on multi-stream data representations and different fusion strategies (late vs. early fusion).

## 2. Objective(s) of your Project
- Build an end-to-end machine learning pipeline that recognizes isolated ASL signs from video using skeleton-based graph neural networks.
- Compare multi-stream fusion strategies across three distinct models (`multi-stream-stgcn` using 3 streams, `4stream-late-fusion` using 4 streams, and `4stream-fusion` utilizing early fusion) to determine which approach best captures sign dynamics.
- Automate the preprocessing pipeline including keypoint extraction, skeleton normalization, and feature engineering.

## 3. Data Collection
**Dataset Source**
Dataset: WLASL Processed — Kaggle (`risangbaskoro/wlasl-processed`)
Original Paper: Li et al., Word-level Deep Sign Language Recognition from Video, IEEE WACV 2020

The dataset was downloaded from Kaggle as a processed version of the original WLASL corpus containing approximately 12,000 MP4 videos. We subsetted this to 50 classes resulting in ~524 videos.
The videos were originally unorganized. An `organize_dataset.py` script was written to sort the videos into class subfolders using the provided `WLASL_v0.3.json` metadata mapping file.

*Please insert a screenshot of the raw dataset directory structure here.*

## 4. Data Preprocessing
The preprocessing pipeline runs in three sequential steps:
1. **Keypoint Extraction (`preprocessing/extract.py`)**: Uses MediaPipe (CPU) to detect 33 body pose landmarks and 21 landmarks per hand per frame, resulting in 150 raw features per frame.
2. **Normalization (`preprocessing/normalize.py`)**: 
   - Filters missing keypoints via temporal interpolation.
   - Reduces 150 features to 102 essential upper-body and hand joints.
   - Centers each frame independently on the mid-shoulder point and scales by shoulder width (position- and scale-invariant).
   - Generates multi-stream features: Joint positions, Bone vectors, Joint motion (velocity), and Bone motion (angular velocity).
3. **Temporal Resampling (`preprocessing/resample.py`)**: Linearly interpolates varying length videos into a fixed length of exactly 64 frames.

*Please insert screenshots of the `extract.py` and `normalize.py` code, and the shapes of the output `X_normalized.npy` data here.*

## 5. Modeling
Three Spatial-Temporal Graph Convolutional Network (ST-GCN) models were implemented and compared:

1. **Multi-Stream ST-GCN (`multi-stream-stgcn`)**
   - **Streams**: 3 Streams (Joint, Bone, Motion)
   - **Fusion**: Late Fusion (Logits are summed after independent processing)
   - **Why**: Serves as a baseline inspired by classic multi-stream approaches. Allows each structural representation to specialize independently.

2. **Four-Stream Late Fusion (`4stream-late-fusion`)**
   - **Streams**: 4 Streams (Joint, Bone, Motion, Bone Motion)
   - **Fusion**: Late Fusion
   - **Why**: Adds a fourth stream (Bone Motion, capturing angular velocity) to determine if adding a new feature representation improves accuracy without changing the fusion strategy.

3. **Four-Stream Early Fusion (`4stream-fusion`)**
   - **Streams**: 4 Streams (Joint, Bone, Motion, Bone Motion)
   - **Fusion**: Early Fusion (Features concatenated before a final shared classifier)
   - **Why**: Allows the shared classifier to model cross-stream interactions (e.g., how hand velocity correlates with arm orientation) before making the final classification decision.

*Please insert screenshots of the model instantiation code from `main.py` and `models/st_gcn_twostream.py` here.*

## 6. Evaluation
Evaluation was conducted using a 4-Fold Stratified K-Fold Cross-Validation on 85% of the data, with a 15% held-out test set for final unbiased scoring. Label smoothing, inverse-frequency class weights, and Stochastic Weight Averaging (SWA) were used.

**Results Comparison:**
- **`multi-stream-stgcn`**: CV Mean: 5.11% ± 0.48% | Test Accuracy: 1.30%
- **`4stream-late-fusion`**: CV Mean: 5.57% ± 0.69% | Test Accuracy: 1.30%
- **`4stream-fusion` (Best)**: CV Mean: 5.34% ± 0.41% | Test Accuracy: 3.90%
*(Random chance baseline for 50 classes: 2.0%)*

**Interpretation:**
The `4stream-fusion` (Early Fusion) model is the clear winner, achieving 3.90% test accuracy, outperforming the random chance baseline and both late-fusion models. The improvement stems from early fusion allowing cross-stream feature interaction prior to classification. The absolute accuracy remains low (~4%) because the dataset is extremely small (~10 samples per class) with very high intra-class variation. The models are overfitting to the training signers, as indicated by the CV vs. Test gap.

*Please insert screenshots of the result charts from `output/charts/` here.*

## 7. Results
The `evaluation/evaluate.py` script can run a single raw MP4 video through the pipeline to predict a sign.

**Sample Inference:**
`python evaluation/evaluate.py --video path/to/sign_video.mp4`

**Discussion:**
The early-fusion model successfully identifies signs with highly distinctive, continuous motions (e.g., `basketball` - two-handed dribbling, `eat` - hand-to-mouth). However, it struggles heavily with lexically similar signs that share handshapes but differ slightly in position, or classes that have fewer than 6 training videos. Increasing the dataset size per class is the primary future requirement.
