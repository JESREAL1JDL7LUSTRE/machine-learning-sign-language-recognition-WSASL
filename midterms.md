1. Overview of the Project
This project develops an Isolated Sign Language Recognition (ISLR) system capable of taking short video clips of American Sign Language (ASL) signs and classifying which sign is being performed. The system operates entirely from skeleton keypoints, it never directly processes raw pixel data. Instead, it extracts the positions of body joints and finger joints from each video frame, preprocesses those coordinates, and feeds them into a deep learning model trained to distinguish between sign classes.

The dataset used is WLASL (World-Level American Sign Language), sourced from Kaggle. The raw dataset consists of approximately 12,000 short video clips covering 2,000 ASL glosses (words), with each video named by a unique ID and organized via a companion WLASL_v0.3.json label file.

The project evolved significantly over its development, from a simple BiLSTM on body-only keypoints, through multiple stages of model improvement, keypoint extraction upgrades, and training methodology changes. This document walks through every major decision and evolution in the order it occurred.
2. Objectives of the Project
The project was designed to achieve the following goals:
Build a pipeline that takes raw sign language videos as input and produces a classification of the sign being performed.
Extract meaningful skeleton keypoints from video using computer vision tools (initially YOLO, then MediaPipe).
Preprocess the raw keypoints into normalized, structured feature representations suitable for deep learning.
Train a deep learning model capable of recognizing signs from skeleton sequences.
Evaluate the model honestly using rigorous methodology to avoid overfitting to the evaluation set.
Iteratively improve accuracy through better features, better model architectures, and better training strategies.
3. Data Collection
3.1 Dataset: WLASL (World-Level American Sign Language)
The raw dataset was downloaded from Kaggle (risangbaskoro/wlasl-processed). After download, the data was structured as a flat directory of approximately 12,000 .mp4 video files named by numeric ID, accompanied by a single WLASL_v0.3.json label file that maps each video ID to its corresponding ASL gloss (word label). No folder structure was provided to indicate class membership.











3.2 Dataset Organization
A preprocessing script (data/organize_dataset.py) was written to parse WLASL_v0.3.json, match each video ID to its gloss label, and copy videos into organized class subfolders under a dataset/ directory. A --subset N argument was used to limit training to the top N most-common glosses, which was critical for keeping processing time manageable:















The organized dataset structure is shown below, with each folder corresponding to one ASL sign class. Experiments began with 20 classes and later expanded to 50 classes. One notable issue encountered was that the raw videos/ folder was inadvertently being picked up as a class, this was resolved by explicitly excluding it in the extraction script.



4. Data Preprocessing
Preprocessing underwent three major evolutionary phases, each motivated by limitations discovered in the previous approach.
4.1 Phase 1: YOLO-Based Extraction (Body Only, 34 Features)
The first extraction attempt used YOLOv8 pose (yolov8n-pose.pt). This was chosen due to a compatibility constraint: the project initially ran on Python 3.13, and MediaPipe only supports up to Python 3.11. YOLO was the practical workaround available.
YOLOv8 extracts 17 body keypoints in COCO format (× 2 coordinates = 34 features per frame). However, it has no dedicated hand landmark detection. For sign language, where hand shape and finger positions carry the majority of the linguistic signal, this representation proved wholly insufficient. The model achieved only 2–4% accuracy regardless of architecture improvements, confirming that the feature set, not the model, was the primary bottleneck.
4.2 Phase 2: YOLO Body + YOLO Hands (118 Features)
To improve hand coverage without abandoning YOLO, a second model dedicated to hand pose estimation was added: yolo11n-pose-hands.pt (21 keypoints per hand, sourced from GitHub). This extended the feature vector to 118 dimensions (34 body + 42 left hand + 42 right hand).
Despite the improved coverage, YOLO-based hand detection remained noisy on close-up sign language footage, where signers' hands often fill the frame in ways that differ from YOLO's general training distribution. Accuracy remained near 2%.
4.3 Phase 3: MediaPipe Tasks API (150 Features, Current Final Approach)
The decisive improvement came from switching to MediaPipe, which provides dedicated, high-accuracy models for both full-body pose estimation and hand landmark detection. To enable MediaPipe, the project was migrated from Python 3.13 to Python 3.11, and the virtual environment was rebuilt accordingly.
Two MediaPipe models were used: pose_landmarker_full.task (33 body keypoints) and hand_landmarker.task (21 keypoints per hand, both hands). This produced a 150-dimensional feature vector per frame: 33 pose × 2 + 21 left hand × 2 + 21 right hand × 2 = 150. Note that MediaPipe GPU acceleration is not supported on Windows, so all extraction was performed on CPU (AMD Ryzen 7 5700X3D), taking approximately 10–15 minutes for 20 classes.























  




4.4 Normalization Pipeline (normalize.py)
Raw keypoints were processed through a seven-step normalization pipeline designed to make the representations invariant to signer position, scale, and camera distance.
Step 1: Missing Keypoint Interpolation
MediaPipe occasionally fails to detect keypoints (e.g., when a hand moves off-screen). Missing values (zeros) are filled via linear interpolation between neighboring frames to produce a continuous signal.
Step 2: Joint Filtering (150 → 102 Features)
Many of the 150 raw features contribute noise rather than signal for ASL recognition. Leg joints (landmarks 25–32) are entirely irrelevant for hand signing; most fine facial details are similarly unnecessary. The pipeline retains only 9 upper-body joints (nose, shoulders, elbows, wrists, hips) plus all 21 left-hand and 21 right-hand landmarks = 51 joints total, yielding 102 features. This reduction improved accuracy noticeably.
Step 3: Centering and Scale Normalization
All coordinates are centered by subtracting the midpoint between the left and right hip, making the skeleton position-invariant (the signer may appear anywhere on screen). Coordinates are then divided by the torso height (mid-hip to nose distance), making the skeleton scale-invariant across signers at different distances from the camera.
Step 4: Relative Hand Coordinates
All finger joint coordinates are expressed relative to their respective wrist. This makes hand shape location-independent, the model learns the configuration of the hand, not its absolute position in the frame.
Step 5: Temporal Smoothing
A Gaussian filter (σ = 1.0) is applied along the time axis for each joint to reduce frame-by-frame jitter inherent in MediaPipe detection.
Steps 6 & 7: Multi-Stream Feature Derivation
From the cleaned and normalized joint stream, three additional feature streams are derived to capture different aspects of signing motion:



 
Stream
Formula
File
Meaning
Joint Positions
Raw normalized coords
X_final.npy
Where joints are
Bone Vectors
Child joint − Parent joint
X_bones.npy
Relative limb orientations
Joint Motion
joint(t) − joint(t−1)
X_motion.npy
How joints move over time
Bone Motion
bone(t) − bone(t−1)
X_bone_motion.npy
How limb orientations change

 





4.5 Resampling (resample.py)
Videos in the WLASL dataset vary in length from approximately 1 to 4 seconds. To produce fixed-length inputs for the neural network, each sequence is resampled to exactly 64 frames using linear interpolation. Short sequences are upsampled and long sequences are downsampled, ensuring all inputs share the same temporal dimension.

 
5. Modeling
The model architecture underwent three distinct generational improvements, driven by growing understanding of the task's spatial-temporal structure.
5.1 Generation 1: BiLSTM (Baseline)
The initial model was a Bidirectional Long Short-Term Memory (BiLSTM) network, a standard baseline for sequential data tasks. The architecture takes the skeleton sequence as a flat feature vector per frame and reads it both forward and backward.
Architecture: Input (batch, 64, features) → BiLSTM (128 hidden, 2 layers) → Dropout (0.5) → Fully Connected (256 → num_classes).
Performance with this model revealed a critical insight: the feature set matters more than the model architecture at this stage. Switching from YOLO body-only features (2% accuracy) to MediaPipe features (26.8% accuracy) represented a larger gain than any model improvement made during this phase.
The fundamental limitation of the BiLSTM approach is that it treats each skeleton frame as a flat vector, with no awareness of spatial relationships between joints. It cannot know that the wrist connects to the elbow, or that adjacent finger joints should move coherently. This structural information is critical for accurate sign classification.
5.2 Generation 2: Custom Multi-Stream ST-GCN
The second architecture was a Spatial-Temporal Graph Convolutional Network (ST-GCN), implemented from scratch following the approach described by Yan et al. (AAAI 2018). ST-GCN addresses the BiLSTM's key limitation by representing the skeleton as a graph where nodes are joints and edges are anatomical connections, allowing the model to propagate information along physiologically meaningful pathways.
Input shape: (batch, 2, 64, 51) — 2 channels (x, y), 64 frames, 51 joints — processed through 9 ST-GCN blocks followed by global average pooling and a fully connected classification head. Three independent ST-GCN branches were used to process joint positions, motion (temporal delta), and bone vectors, with their output logits summed (late fusion) before the final softmax.
This architecture produced a test accuracy of 45.2% and a cross-validation mean of 62.4%, representing a substantial improvement over the BiLSTM baseline of 26.8%. 
5.3 Generation 3: Ported Original ST-GCN with Enhancements (Current)
The final architecture ports the original ST-GCN implementation by Yan et al. (2018) to modern PyTorch, adding four significant enhancements informed by subsequent literature.
Enhancement 1: Bone Motion Stream (4th Stream)
Based on Miah et al. (2023), the temporal derivative of bone vectors (bone motion) was added as a fourth input stream. This stream distinguishes signs that share similar joint configurations but differ in movement speed or direction.
Enhancement 2: Early Fusion
In the previous generation, streams were combined by summing their final classification logits (late fusion), which prevented the model from learning cross-stream feature interactions. The current model instead concatenates 256-dimensional feature vectors from each stream into a 1024-dimensional representation, then passes this through a shared classification head, enabling the model to discover richer multi-stream relationships.
Enhancement 3: Adaptive Graph Topology
Inspired by Zhang et al. STA-GCN (2020), a learnable adjacency matrix is added on top of the fixed anatomical graph. This allows the model to discover non-anatomical joint correlations — for example, both hands coordinating together in two-handed signs — that the fixed skeleton topology cannot represent.
Enhancement 4: DropGraph Regularization
Following Jiang et al. (2021), DropGraph randomly zeros entire joint channels during training, forcing the model to avoid relying on any single joint. This improves robustness to MediaPipe detection failures, which occasionally occur when hands partially leave the frame.
The full architecture is summarized below:
 










 



6. Evaluation
6.1 Evaluation Methodology
Phase 1: Simple Train/Validation/Test Split
Initially the dataset was partitioned 70/15/15 (train/validation/test). With only 201 samples across 20 classes, this approach proved fundamentally unreliable: different random seeds could shift test accuracy by 5–10 percentage points simply by placing harder samples in the test partition. Stratification was also required to ensure every class was represented across all splits.
Phase 2: Stratified K-Fold Cross-Validation (Current)
The evaluation was upgraded to Stratified K-Fold Cross-Validation (K=4). A fixed test set of 31 samples was held out from all folds. The remaining data was divided into 4 folds; in each iteration, the model was trained on 3 folds and validated on the 4th. K=4 was chosen over K=5 because some classes contain only 5–6 samples — with K=5, the stratification constraint becomes impossible to satisfy for minority classes.
The primary reported metric is CV Mean ± Standard Deviation across all folds (e.g., 0.624 ± 0.043), supplemented by the final test accuracy on the held-out set. The CV Mean is the more trustworthy figure: with only 31 test samples, a single misclassification shifts test accuracy by approximately 3 percentage points, whereas the cross-validation estimate averages over four independent validation sets.



6.2 Training Improvements
The following improvements were applied incrementally to stabilize and improve training:

Improvement
Rationale
Early Stopping (patience=20–35)
Prevents wasting epochs after validation plateau; reduces overfitting
Warmup + Cosine LR Schedule
Gentler learning rate start (10 warmup epochs), then smooth decay
AdamW Optimizer
Decoupled weight decay for more effective regularization
Label Smoothing (0.1–0.15)
Prevents overconfident predictions; uses soft targets instead of hard 1/0 labels
Weighted Loss
Inverse-frequency class weights compensate for class imbalance
Gradient Clipping (1.0)
Prevents exploding gradients on small batch sizes


6.3 Data Augmentation
The following augmentations were applied per sample during training to improve generalization:
Augmentation
Effect
Gaussian Noise (σ=0.015)
Simulates natural jitter in MediaPipe keypoint detection
Random Scale (0.8–1.2×)
Simulates signers at different distances from the camera
Random Rotation (±20°)
Simulates minor variations in camera angle
Horizontal Flip
Doubles effective data size; models handedness invariance
Random Frame Drop (15%)
Simulates variable signing speed and partial occlusion
Time Warp
Stretches or compresses the temporal sequence non-linearly
Temporal Crop (75–100%)
Uses a random sub-window of the full sequence

6.4 Overfitting Diagnosis and Response
A persistent challenge throughout training was the gap between training and validation accuracy. The table below shows the evolution of this diagnostic across key runs:
Run
Train Acc.
Val Acc.
Gap
Status
Early runs
93%
45%
48%
Severe overfitting
After aggressive regularization
35%
37%
~0%
Underfitting
Target sweet spot
~65%
~55%
~10%
Balanced


Severe overfitting was caused by approximately 13 million model parameters being trained on only 136 training samples per fold. Regularization strategies applied in response included increasing dropout from 0.4 to 0.6, increasing weight decay from 1e-3 to 5e-2, adding DropGraph regularization, and enforcing early stopping. When regularization was overly aggressive, the model underfitted; the optimal balance was found at dropout=0.5–0.6 with weight_decay=1e-2–5e-2.

6.5 Results Summary
The table below summarizes all model versions and their corresponding accuracy results. The current best result is highlighted.
 
Version
Model
Features
Test Acc.
CV Mean
BiLSTM (body only)
BiLSTM
34
2%
—
BiLSTM + YOLO Hands
BiLSTM
118
2%
—
BiLSTM + MediaPipe
BiLSTM
150
26.8%
—
Custom ST-GCN (filtered joints)
ST-GCN
102
33.3%
—
Multi-stream ST-GCN (3-stream)
Custom ST-GCN ×3
102×3
45.2%
62.4%
Original 2-stream ST-GCN (ported)
Yan et al. ×2
102×2
38.7%
51.8%
4-stream Early Fusion (current)
Yan et al. ×4 + Enhancements
102×4
48.4%
51.2%

Current best: 48.4% test accuracy / 51.2% CV Mean on 50 classes.
Feature quality beats model complexity. The switch from YOLO (2%) to MediaPipe (26.8%) was the single largest accuracy gain in the entire project, larger than any model improvement. A better feature set outweighs a more sophisticated architecture when the input representation is poor.
Data scarcity is the true bottleneck. With 6–16 samples per class, even a near-optimal model cannot generalize reliably. The literature recommends 50–100+ samples per class for stable training; the current 20-class experiment averages approximately 10 per class.
Multi-stream input is highly effective. Each additional feature stream (bone, motion, bone motion) contributed a meaningful signal: expanding from 1 stream to 3 streams improved test accuracy from 33.3% to 45.2%.
Late fusion vs. early fusion matters. Summing final classification logits (late fusion) prevents the model from learning cross-stream interactions. Early fusion via feature concatenation allows the fusion head to discover richer joint representations across streams.
K-Fold cross-validation gives a more honest picture. A single train/test split on 201 samples is inherently noisy. K-Fold reveals not only average performance but also variance (±0.05), exposing when a model is unstable across different data partitions.
More data is the next priority. The pipeline is now mature. Expanding from 20 to 50 classes gives approximately 500 total samples (~25 per class), which is where meaningful generalization begins to become achievable.

REFERENCES
Yan, S., Xiong, Y., & Lin, D. (2018). Spatial temporal graph convolutional networks for skeleton-based action recognition. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), 7444–7452. https://doi.org/10.1609/aaai.v32i1.12328
Miah, A. S. M., Hasan, M. A. M., Shin, J., Okuyama, Y., & Tomioka, Y. (2023). Multistream spatial-temporal graph convolutional network for skeleton-based sign language recognition. Sensors, 23(3), 1711. https://doi.org/10.3390/s23031711
Zhang, W., Lin, Z., Cheng, J., Ma, C., Deng, X., & Wang, H. (2020). STA-GCN: Two-stream graph convolutional network with spatial–temporal attention for hand gesture recognition. The Visual Computer, 36(10–12), 2433–2444. https://doi.org/10.1007/s00371-020-01955-w
Jiang, S., Sun, B., Wang, L., Bai, Y., Li, K., & Fu, Y. (2021). Skeleton Aware Multi-modal Sign Language Recognition (arXiv:2103.08833). arXiv. https://doi.org/10.48550/arXiv.2103.08833
GitHub Repository Link:
https://github.com/JESREAL1JDL7LUSTRE/machine-learning-sign-language-recognition-WSASL.git
STA-GCN Repository:
https://github.com/yysijie/st-gcn.git
Datasets:
https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed?resource=download&select=nslt_100.json
