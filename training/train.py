import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# ── Path fix so we can import from sibling folders ────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.lstm      import SignLSTM
from training.dataset import SignDataset

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = os.path.join(ROOT, "output")
MODEL_SAVE   = os.path.join(ROOT, "models", "sign_lstm.pth")

BATCH_SIZE   = 16
EPOCHS       = 30
LEARNING_RATE= 0.001
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
BIDIRECTIONAL= True
TEST_SPLIT   = 0.2   # 80 % train, 20 % validation
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Load Data ─────────────────────────────────────────────────────────────────
def load_data():
    # Prefer fully preprocessed data; fall back gracefully
    for fname in ["X_final.npy", "X_normalized.npy", "X_raw.npy"]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            print(f"Loading {fname} ...")
            X = np.load(path)
            break
    else:
        raise FileNotFoundError(
            "No data found in output/. Run preprocessing scripts first.\n"
            "  1. preprocessing/extract.py\n"
            "  2. preprocessing/normalize.py\n"
            "  3. preprocessing/resample.py"
        )

    y_path = os.path.join(OUTPUT_DIR, "y.npy")
    if not os.path.exists(y_path):
        raise FileNotFoundError("y.npy not found in output/. Run extract.py first.")

    y = np.load(y_path)
    print(f"   X shape : {X.shape}   y shape : {y.shape}")
    return X, y


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    X, y = load_data()

    num_classes = len(np.unique(y))
    print(f"   Classes : {num_classes}")

    # Train / validation split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size   = TEST_SPLIT,
        random_state= SEED,
        stratify    = y
    )

    train_set = SignDataset(X_train, y_train, augment=True)
    val_set   = SignDataset(X_val,   y_val,   augment=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = SignLSTM(
        input_size   = X.shape[2],       # 150
        hidden_size  = HIDDEN_SIZE,
        num_layers   = NUM_LAYERS,
        num_classes  = num_classes,
        dropout      = DROPOUT,
        bidirectional= BIDIRECTIONAL
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler — reduce LR when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=True
    )

    best_val_acc = 0.0

    print(f"\n{'='*55}")
    print(f"  Training  |  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}")
    print(f"{'='*55}")

    for epoch in range(1, EPOCHS + 1):

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            train_correct += (output.argmax(1) == y_batch).sum().item()

        train_acc  = train_correct / len(train_set)
        train_loss /= len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                output    = model(X_batch)
                val_loss += criterion(output, y_batch).item()
                val_correct += (output.argmax(1) == y_batch).sum().item()

        val_acc  = val_correct / len(val_set)
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE)
            marker = "  ← best"
        else:
            marker = ""

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}{marker}"
        )

    print(f"\n✅ Training complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"   Model saved → {MODEL_SAVE}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train()