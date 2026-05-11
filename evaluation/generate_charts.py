import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.gridspec as gridspec
import re

plt.switch_backend("Agg")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT, "output")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

MODEL_FILES = {
    "Multi-stream ST-GCN (3-stream)": "results_3stream.json",
    "4-stream Late Fusion": "results_4stream_late.json",
    "4-stream Early Fusion": "results_4stream_early.json"
}

def load_results():
    results = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            with open(path, 'r') as f:
                results[name] = json.load(f)
        else:
            print(f"Warning: {fname} not found.")
    return results

def plot_comparison(results):
    if not results:
        print("No results to plot.")
        return

    names = list(results.keys())
    test_accs = [results[n]["test_acc"] * 100 for n in names]
    cv_means = [results[n]["cv_mean"] * 100 for n in names]
    cv_stds = [results[n]["cv_std"] * 100 for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, cv_means, width, yerr=cv_stds, label='CV Mean Acc (%)', capsize=5, color='#4C72B0')
    rects2 = ax.bar(x + width / 2, test_accs, width, label='Test Acc (%)', color='#55A868')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, 100)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "comparison_overview.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def _safe_name(name):
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def plot_model_results(name, results):
    """Create a 4-panel figure similar to main.py's `plot_model_results`.
    Saves to CHARTS_DIR/<safe_name>_results.png
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    color = '#55A868'
    fold_accs = results.get('fold_accs', [])
    histories = results.get('fold_histories', [])
    n_cls = int(results.get('num_classes', 0))

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Fold accuracies
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    bars = ax1.bar([f"Fold {i+1}" for i in range(len(fold_accs))], fold_accs, color=color, alpha=0.85, width=0.55, zorder=3)
    cv_mean = results.get('cv_mean', 0.0)
    test_acc = results.get('test_acc', 0.0)
    ax1.axhline(cv_mean, color=color, ls='--', lw=1.5, label=f"CV Mean {cv_mean:.3f}")
    ax1.axhline(test_acc, color="#333333", ls=':', lw=1.5, label=f"Test Acc {test_acc:.3f}")
    for bar, acc in zip(bars, fold_accs):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha='center', va='bottom', fontsize=8)
    ax1.set_ylim(0, min(1.0, max(fold_accs) * 1.25 + 0.05) if fold_accs else 1.0)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("K-Fold Validation Accuracy", fontweight="bold")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # Panel 2: Training curves
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    for i, h in enumerate(histories):
        t = h.get('train_acc', [])
        v = h.get('val_acc', [])
        if t:
            ax2.plot(t, linestyle='--', alpha=0.6, label=f'Fold {i+1} Train')
        if v:
            ax2.plot(v, alpha=0.8, label=f'Fold {i+1} Val')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Curves (all folds)\n─ val   ╌ train", fontweight="bold")
    ax2.grid(alpha=0.25)
    ax2.set_ylim(0, 1.0)

    # Panel 3: Confusion matrix (test)
    ax3 = fig.add_subplot(gs[0, 2])
    all_preds = np.array(results.get('all_preds', []))
    all_labels = np.array(results.get('all_labels', []))
    if len(all_preds) and len(all_labels) and n_cls > 0:
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_cls)))
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax3.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        if n_cls <= 25:
            label_map = results.get('label_map', {})
            idx_to_label = {v: k for k, v in label_map.items()} if label_map else {}
            ticks = [idx_to_label.get(i, str(i)) for i in range(n_cls)]
            ax3.set_xticks(range(n_cls)); ax3.set_yticks(range(n_cls))
            ax3.set_xticklabels(ticks, rotation=90, fontsize=6)
            ax3.set_yticklabels(ticks, rotation=0, fontsize=6)
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("True")
        ax3.set_title(f"Confusion Matrix (Test Set, n={len(all_labels)})", fontweight="bold")
    else:
        ax3.text(0.5, 0.5, "No test predictions", ha='center', va='center')
        ax3.axis('off')

    # Panel 4: Per-class accuracy
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor("#FFFFFF")
    per_src_preds = np.array(results.get('cv_preds', results.get('all_preds', [])))
    per_src_labels = np.array(results.get('cv_labels', results.get('all_labels', [])))
    label_map = results.get('label_map', {})
    idx_to_label = {v: k for k, v in label_map.items()} if label_map else {}
    counts = np.bincount(per_src_labels, minlength=n_cls) if len(per_src_labels) else np.zeros(n_cls, dtype=int)
    present = counts > 0
    per_class_acc = np.full(n_cls, np.nan, dtype=np.float32)
    for cls_id in range(n_cls):
        if present[cls_id]:
            mask = per_src_labels == cls_id
            per_class_acc[cls_id] = (per_src_preds[mask] == per_src_labels[mask]).mean() if mask.any() else np.nan
    cls_labels = [idx_to_label.get(i, str(i)) for i in range(n_cls)]
    mean_acc = float(np.nanmean(per_class_acc)) if np.any(present) else 0.0
    plot_vals = np.where(np.isnan(per_class_acc), 0.0, per_class_acc)
    bar_colors = [color if (not np.isnan(a) and a >= mean_acc) else "#CCCCCC" for a in per_class_acc]
    ax4.bar(range(n_cls), plot_vals, color=bar_colors, alpha=0.85)
    missing_count = int((~present).sum()) if n_cls else 0
    ax4.axhline(mean_acc, color="#333333", ls="--", lw=1.5, label=f"Mean {mean_acc:.3f} (present only)")
    if missing_count:
        ax4.text(0.99, 0.02, f"{missing_count} missing classes", transform=ax4.transAxes, ha='right', va='bottom', fontsize=8)
    ax4.set_xticks(range(n_cls))
    ax4.set_xticklabels(cls_labels, rotation=90, fontsize=6 if n_cls > 30 else 8)
    ax4.set_ylabel("Accuracy")
    src_name = "CV" if 'cv_preds' in results else "Test"
    ax4.set_title(f"Per-Class Accuracy ({src_name})", fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.set_ylim(0, 1.15)
    ax4.grid(axis="y", alpha=0.25)

    # Panel 5: Summary box
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#FFFFFF")
    ax5.axis("off")
    lines = [
        f"Model: {name}",
        f"Classes: {n_cls}",
        f"CV Mean: {results.get('cv_mean', 0.0):.3f}",
        f"CV Std : {results.get('cv_std', 0.0):.3f}",
        f"Test Acc: {results.get('test_acc', 0.0):.3f}",
    ]
    ax5.text(0, 1, "\n".join(lines), va='top', fontsize=10)

    fig.tight_layout()
    fname = f"{_safe_name(name)}_results.png"
    out_path = os.path.join(CHARTS_DIR, fname)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

def plot_confusion_matrices(results):
    for name, res in results.items():
        y_true = res.get("all_labels", [])
        y_pred = res.get("all_preds", [])
        if not y_true or not y_pred:
            continue
            
        lmap = res.get("label_map", {})
        classes = sorted(lmap.keys(), key=lambda k: lmap[k])
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize by row
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

        plt.figure(figsize=(14, 12))
        sns.heatmap(cm_norm, cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Normalized Confusion Matrix: {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        out_path = os.path.join(CHARTS_DIR, f"cm_{safe_name}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")

def plot_histories(results):
    for name, res in results.items():
        histories = res.get("fold_histories", [])
        if not histories:
            continue
            
        plt.figure(figsize=(10, 6))
        for fold, hist in enumerate(histories, 1):
            t_acc = hist.get("train_acc", [])
            v_acc = hist.get("val_acc", [])
            plt.plot(t_acc, label=f'Fold {fold} Train', linestyle='--', alpha=0.6)
            plt.plot(v_acc, label=f'Fold {fold} Val', alpha=0.8)
            
        plt.title(f'Training History: {name}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        # plt.legend() # Legend gets too crowded with 4 folds * 2 lines
        
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        out_path = os.path.join(CHARTS_DIR, f"history_{safe_name}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    results = load_results()
    if results:
        plot_comparison(results)
        plot_confusion_matrices(results)
        plot_histories(results)
    else:
        print("No JSON results found. Please run the training scripts first.")
