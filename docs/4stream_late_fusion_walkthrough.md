# Walkthrough: 4-Stream Late Fusion Model

## Objective
This walkthrough explains the `4stream-late-fusion` model line by line, demonstrating how independent stream processing works in the codebase.

## 1. Instantiation (`main.py`)
```python
def run_4stream_late_fusion(X_joint, X_bone, X_motion, X_bm, y, epochs, device, label_map):
    """4-stream late fusion ST-GCN."""
    num_classes = len(np.unique(y))

    def make_model():
        return FourStreamSTGCN(
            2, num_classes, GRAPH_ARGS,
            early_fusion=False,  # <-- Crucial parameter
            dropout=DROPOUT,
        ).to(device)
```
The model is constructed exactly like the winning `4stream-fusion` model, except `early_fusion` is explicitly set to `False`.

## 2. Forward Pass (`models/st_gcn_twostream.py`)
Inside the `FourStreamSTGCN` model definition, the forward pass behaves differently depending on the fusion flag:

```python
    def forward(self, x, motion=None, bone=None, bone_motion=None):
        # In Late Fusion, extract_features is False for the sub-models.
        # This means they output logits directly, not feature vectors.
        
        logits_joint = self.joint_stream(x)
        logits_motion = self.motion_stream(motion)
        logits_bone = self.bone_stream(bone)
        logits_bone_motion = self.bone_motion_stream(bone_motion)

        if not self.early_fusion:
            # Late Fusion: Sum the independent logits
            return logits_joint + logits_motion + logits_bone + logits_bone_motion
```

## 3. Why it matters
Because the final output is just a sum, the model cannot learn complex relationships like "if the hand is moving fast (Motion Stream) AND the arm is pointing up (Bone Stream)". The early fusion variant concatenates the raw feature vectors and passes them through a dense linear layer, enabling the network to learn these crucial cross-stream interactions.
