# 2-Stream ST-GCN Walkthrough (Beginner, Step-by-Step)

This explains every part of the 2-stream model in very simple language. The main code is in `models/stgcn_2stream_ported.py`, and it uses the backbone in `models/st_gcn.py`.

## 1) Big idea in one sentence
We use two separate networks to look at the same sign: one uses joints, the other uses bones. Each network makes a prediction, then we add those predictions together.

## 2) What "stream" means
A **stream** is just one copy of the same model that gets a different kind of input:
- Stream A: joint positions (where each joint is in the frame).
- Stream B: bone vectors (the direction and length from parent joint to child joint).

## 3) Input and output (simple)
- Input `x`: `(N, 2, T, V)` where:
  - `N` = batch size
  - `2` = x and y coordinates
  - `T` = number of frames (64 after resampling)
  - `V` = number of joints (51)
- Input `bone`: same shape, but each joint holds a bone vector.
- Output: `(N, num_classes)` raw scores (logits).

## 4) File: `models/stgcn_2stream_ported.py`

### 4.1) `TwoStreamPortedSTGCN.__init__`
**What it does:**
- Creates two copies of the `ST_GCN` model.
- One copy will process joints.
- The other will process bones.

**Important settings:**
- `graph_args` sets the skeleton layout (`mediapipe_51`) and graph strategy (`spatial`).
- `adaptive_graph=True` means the graph can learn extra connections.
- `drop_graph_prob=0.1` randomly drops joints during training.

### 4.2) `TwoStreamPortedSTGCN.forward`
**What it does:**
1. If `bone` is missing, create a zero array (so the code still works).
2. Send joints through the joint stream.
3. Send bones through the bone stream.
4. Add the two outputs together.
5. Return the final logits.

**Why add the logits?**
This is called **late fusion**. Each stream is allowed to make its own decision, and then their opinions are combined at the end.

## 5) File: `models/st_gcn.py` (the backbone)
This is the core ST-GCN network used by each stream.

### 5.1) Graph setup
- `Graph(**graph_args)` builds a skeleton graph (nodes = joints, edges = bones).
- `A` is the adjacency matrix: it tells which joints are connected.

### 5.2) Adaptive graph
- `AdaptiveAdjacency` creates extra learnable connections.
- The final graph becomes: `A_fixed + alpha * A_learned`.
- This lets the model discover useful connections not in the original skeleton.

### 5.3) DropGraph
- `drop_graph(x, prob)` randomly removes joints during training.
- This helps the model be robust to missing keypoints.

### 5.4) ST-GCN layers
- `self.st_gcn_networks` is a list of 10 blocks.
- Each block mixes information across joints (graph convolution) and across time (temporal convolution).
- Channels are smaller than the original paper (32/64/128) to reduce overfitting.

### 5.5) Final head
- Global average pooling collapses time and joints.
- If `extract_features=False`, it outputs class logits.
- If `extract_features=True`, it outputs features (used in early fusion models).

## 6) File: `models/graph.py` (graph building)
- Defines the 51-joint layout and how joints connect.
- Builds adjacency matrices using a "spatial" partitioning strategy.
- Offers optional `AdaptiveAdjacency` and `drop_graph`.

## 7) File: `models/tgcn.py` (graph convolution)
- Runs a temporal convolution first (across time).
- Then multiplies by the adjacency matrix to share info between joints.
- This is the main "graph convolution" step.

## 8) Where this model is used
- Training runner: `run_2stream_stgcn` in `main.py`.
- It builds the model, then trains with K-fold validation.
- It uses the `TwoStreamDataset` class in `main.py` to feed joint and bone streams.

## 9) Mental picture of one forward pass
1. The video keypoints are shaped as `(2, T, V)`.
2. The joint stream processes them with 10 ST-GCN blocks.
3. The bone stream processes bone vectors the same way.
4. Both streams output scores for each class.
5. The scores are added together to make the final prediction.

## 10) Why this design works
- Joints tell **where** the body is.
- Bones tell **how** joints move relative to each other.
- Combining them gives a more complete picture of the sign.
