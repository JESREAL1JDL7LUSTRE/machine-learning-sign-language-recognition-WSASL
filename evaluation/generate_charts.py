import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import matplotlib.gridspec as gridspec
import re

plt.switch_backend("Agg")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT, "output")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

MODEL_FILES = {
    # keys match main.py MODEL_ORDER / MODEL_RUNNERS
    "multi-stream-stgcn": "results_3stream.json",
    "4stream-late-fusion": "results_4stream_late.json",
    "4stream-fusion": "results_4stream_early.json",
}

def load_results():
    results = {}
    for key, fname in MODEL_FILES.items():
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            with open(path, 'r') as f:
                results[key] = json.load(f)
        else:
            print(f"Warning: {fname} not found.")
    return results

def plot_comparison_overview(all_results, save_path):
    """Generate the comparison overview figure identical to main.py's output."""
    # Use the same plotting style and layout as main.py
    COLORS = {
        "multi-stream-stgcn": "#55A868",
        "2stream-stgcn":      "#C44E52",
        "4stream-fusion":     "#8172B2",
        "4stream-late-fusion":"#D55E00",
    }

    FULL_NAMES = {
        "multi-stream-stgcn": "Multi-Stream ST-GCN\n(3-Stream)",
        "2stream-stgcn":      "Original 2-Stream\nST-GCN (Ported)",
        "4stream-fusion":     "4-Stream Early\nFusion (Current)",
        "4stream-late-fusion":"4-Stream Late\nFusion",
    }

    PUBLISHED = {
        "multi-stream-stgcn": {"test": 0.452, "cv": 0.624, "label": "Multi-Stream\nST-GCN (3s)"},
        "2stream-stgcn":      {"test": 0.387, "cv": 0.518, "label": "2-Stream\nST-GCN (ported)"},
        "4stream-fusion":     {"test": 0.484, "cv": 0.512, "label": "4-Stream\nEarly Fusion"},
        "4stream-late-fusion":{"test": None,  "cv": None,  "label": "4-Stream\nLate Fusion"},
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    keys = [k for k in ["multi-stream-stgcn", "2stream-stgcn", "4stream-fusion", "4stream-late-fusion"] if k in all_results]
    names = [FULL_NAMES[k].replace("\n", " ") for k in keys]
    colors = [COLORS.get(k, "#333333") for k in keys]

    run_test = [all_results[k]["test_acc"] for k in keys]
    run_cv = [all_results[k]["cv_mean"] for k in keys]
    run_cv_std = [all_results[k]["cv_std"] for k in keys]
    pub_test = [PUBLISHED[k]["test"] or 0 for k in keys]
    pub_cv = [PUBLISHED[k]["cv"] or 0 for k in keys]

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    x = np.arange(len(keys))
    w = 0.32

    # Panel 1
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#FFFFFF")
    b1 = ax1.bar(x - w/2, run_test, w, color=colors, alpha=0.85, label="This Run")
    b2 = ax1.bar(x + w/2, pub_test, w, color=colors, alpha=0.40, label="Published")
    for bar, val in list(zip(b1, run_test)) + list(zip(b2, pub_test)):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.003, f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Test Accuracy: This Run vs Published Results", fontweight="bold", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, min(1.0, max(run_test + pub_test) * 1.2 + 0.05))
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#FFFFFF")
    valid_cv = [(i, k) for i, k in enumerate(keys) if all_results[k]["cv_mean"] > 0]
    if valid_cv:
        vi, vk = zip(*valid_cv)
        vcv = [all_results[k]["cv_mean"] for k in vk]
        vstd = [all_results[k]["cv_std"] for k in vk]
        vcolors = [COLORS.get(k, "#333333") for k in vk]
        ax2.bar(vi, vcv, color=vcolors, alpha=0.85, width=0.55)
        ax2.errorbar(vi, vcv, yerr=vstd, fmt="none", color="#333333", capsize=5, lw=1.5)
        pub_cvs = [PUBLISHED[k]["cv"] for k in vk if PUBLISHED[k]["cv"] is not None]
        pub_vi = [i for i, k in zip(vi, vk) if PUBLISHED[k]["cv"] is not None]
        if pub_cvs:
            ax2.scatter(pub_vi, pub_cvs, marker="D", color="#333333", zorder=5, s=50, label="Published CV")
        for i, (cv, std) in zip(vi, zip(vcv, vstd)):
            ax2.text(i, cv + std + 0.005, f"{cv:.3f}±{std:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels(names, fontsize=8, rotation=10)
    ax2.set_ylabel("CV Mean Accuracy")
    ax2.set_title("Cross-Validation Mean ± Std", fontweight="bold")
    ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#FFFFFF")
    ax3.plot(range(len(keys)), run_test, "o-", color="#2196F3", lw=2, ms=8, label="Test (this run)")
    ax3.plot(range(len(keys)), pub_test, "s--", color="#2196F3", lw=1.5, ms=8, alpha=0.5, label="Test (published)")
    ax3.plot(range(len(keys)), run_cv, "o-", color="#4CAF50", lw=2, ms=8, label="CV (this run)")
    if any(p for p in pub_cv):
        ax3.plot(range(len(keys)), pub_cv, "s--", color="#4CAF50", lw=1.5, ms=8, alpha=0.5, label="CV (published)")
    ax3.fill_between(range(len(keys)), [c - s for c, s in zip(run_cv, run_cv_std)], [c + s for c, s in zip(run_cv, run_cv_std)], alpha=0.15, color="#4CAF50")
    ax3.set_xticks(range(len(keys)))
    ax3.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Model Evolution Timeline", fontweight="bold")
    ax3.set_ylim(0, 1.0)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)

    # Panel 4
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#FFFFFF")
    fold_data = [all_results[k]["fold_accs"] for k in keys]
    bp = ax4.boxplot(fold_data, patch_artist=True, widths=0.5, medianprops=dict(color="white", lw=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for whisker in bp["whiskers"]:
        whisker.set(color="#555555", lw=1)
    for cap in bp["caps"]:
        cap.set(color="#555555", lw=1)
    for flier in bp["fliers"]:
        flier.set(marker="o", markerfacecolor="#555555", markersize=4)
    ax4.set_xticks(range(1, len(keys) + 1))
    ax4.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax4.set_ylabel("Fold Accuracy")
    ax4.set_title("Fold Accuracy Distribution", fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    # Panel 5
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#FFFFFF")
    ax5.axis("off")
    table_data = [["Model", "Test\n(run)", "Test\n(pub)", "CV Mean\n(run)", "CV\n(pub)", "Δ Test"]]
    for k in keys:
        r = all_results[k]
        p = PUBLISHED[k]
        delta = r["test_acc"] - (p["test"] or 0)
        table_data.append([
            FULL_NAMES[k].replace("\n", " "),
            f"{r['test_acc']:.3f}",
            f"{p['test']:.3f}" if p["test"] else "—",
            f"{r['cv_mean']:.3f}±{r['cv_std']:.3f}",
            f"{p['cv']:.3f}" if p["cv"] else "—",
            f"{delta:+.3f}",
        ])
    tbl = ax5.table(cellText=table_data[1:], colLabels=table_data[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2196F3")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EEF4FF")
        else:
            cell.set_facecolor("#FFFFFF")
    ax5.set_title("Results Summary Table", fontweight="bold", pad=12)

    fig.suptitle(f"ASL Sign Language Recognition — {len(keys)}-Model Comparison Overview", fontsize=15, fontweight="bold", y=0.99)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved comparison chart → {save_path}")


def _safe_name(name):
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def plot_model_results(key, results, save_path):
    """Generate a 4-panel per-model results figure."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pub = {
        "multi-stream-stgcn": {"test": 0.452, "cv": 0.624, "label": "Multi-Stream\nST-GCN (3s)"},
        "2stream-stgcn":      {"test": 0.387, "cv": 0.518, "label": "2-Stream\nST-GCN (ported)"},
        "4stream-fusion":     {"test": 0.484, "cv": 0.512, "label": "4-Stream\nEarly Fusion"},
        "4stream-late-fusion":{"test": None,  "cv": None,  "label": "4-Stream\nLate Fusion"},
    }[key]
    COLORS = {
        "multi-stream-stgcn": "#55A868",
        "2stream-stgcn":      "#C44E52",
        "4stream-fusion":     "#8172B2",
        "4stream-late-fusion":"#D55E00",
    }
    FULL_NAMES = {
        "multi-stream-stgcn": "Multi-Stream ST-GCN\n(3-Stream)",
        "2stream-stgcn":      "Original 2-Stream\nST-GCN (Ported)",
        "4stream-fusion":     "4-Stream Early\nFusion (Current)",
        "4stream-late-fusion":"4-Stream Late\nFusion",
    }

    color = COLORS.get(key, "#55A868")
    n_cls = results.get("num_classes", 0)
    fold_accs = results.get("fold_accs", [])
    histories = results.get("fold_histories", [])

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    bars = ax1.bar([f"Fold {i+1}" for i in range(len(fold_accs))], fold_accs, color=color, alpha=0.85, width=0.55, zorder=3)
    ax1.axhline(results["cv_mean"], color=color, ls="--", lw=1.5, label=f"CV Mean {results['cv_mean']:.3f}")
    ax1.axhline(results["test_acc"], color="#333333", ls=":", lw=1.5, label=f"Test Acc {results['test_acc']:.3f}")
    if pub["cv"] is not None:
        ax1.axhline(pub["cv"], color=color, ls="-.", lw=1.2, alpha=0.5, label=f"Published CV {pub['cv']:.3f}")
    if pub["test"] is not None:
        ax1.axhline(pub["test"], color="#333333", ls="-.", lw=1.2, alpha=0.5, label=f"Published Test {pub['test']:.3f}")
    for bar, acc in zip(bars, fold_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, acc + 0.005, f"{acc:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, min(1.0, max(fold_accs) * 1.25 + 0.05) if fold_accs else 1.0)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("K-Fold Validation Accuracy", fontweight="bold")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # Panel 2
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    for i, h in enumerate(histories):
        ep = range(1, len(h.get("train_acc", [])) + 1)
        ax2.plot(ep, h.get("train_acc", []), color=color, alpha=0.3, lw=1)
        ax2.plot(ep, h.get("val_acc", []), color=color, alpha=0.7, lw=1.5, label=f"Fold {i+1} val" if i == 0 else "_")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Curves (all folds)\n─ val   ╌ train", fontweight="bold")
    ax2.grid(alpha=0.25)
    ax2.set_ylim(0, 1.0)

    # Panel 3
    ax3 = fig.add_subplot(gs[0, 2])
    all_preds = np.array(results.get("all_preds", []))
    all_labels = np.array(results.get("all_labels", []))
    if len(all_preds) and len(all_labels) and n_cls > 0:
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_cls)))
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax3.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        if n_cls <= 25:
            label_map = results.get("label_map", {})
            idx_to_label = {v: k for k, v in label_map.items()} if label_map else {}
            tick_labels = [idx_to_label.get(i, str(i)) for i in range(n_cls)]
            ax3.set_xticks(range(n_cls))
            ax3.set_yticks(range(n_cls))
            ax3.set_xticklabels(tick_labels, rotation=90, fontsize=6)
            ax3.set_yticklabels(tick_labels, fontsize=6)
    else:
        ax3.text(0.5, 0.5, "No test predictions", ha='center', va='center')
        ax3.axis('off')
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("True")
    ax3.set_title(f"Confusion Matrix (Test Set, n={len(all_labels)})", fontweight="bold")

    # Panel 4
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor("#FFFFFF")
    per_src_preds = np.array(results.get("cv_preds", all_preds))
    per_src_labels = np.array(results.get("cv_labels", all_labels))
    label_map = results.get("label_map", {})
    idx_to_label = {v: k for k, v in label_map.items()} if label_map else {}
    counts = np.bincount(per_src_labels, minlength=n_cls) if len(per_src_labels) else np.zeros(n_cls, dtype=int)
    present = counts > 0
    per_class_acc = np.full(n_cls, np.nan, dtype=np.float32)
    for cls_id in range(n_cls):
        if present[cls_id]:
            mask = per_src_labels == cls_id
            per_class_acc[cls_id] = (per_src_preds[mask] == per_src_labels[mask]).mean()
    cls_labels = [idx_to_label.get(i, str(i)) for i in range(n_cls)]
    mean_acc = float(np.nanmean(per_class_acc)) if np.any(present) else 0.0
    plot_vals = np.where(np.isnan(per_class_acc), 0.0, per_class_acc)
    bar_colors = [color if (not np.isnan(a) and a >= mean_acc) else "#CCCCCC" for a in per_class_acc]
    ax4.bar(range(n_cls), plot_vals, color=bar_colors, alpha=0.85)
    missing_count = int((~present).sum())
    ax4.axhline(mean_acc, color="#333333", ls="--", lw=1.5, label=f"Mean {mean_acc:.3f} (present only)")
    if missing_count:
        ax4.text(0.99, 0.97, f"Missing classes: {missing_count}", transform=ax4.transAxes, ha="right", va="top", fontsize=8, color="#444444")
    ax4.set_xticks(range(n_cls))
    ax4.set_xticklabels(cls_labels, rotation=90, fontsize=6 if n_cls > 30 else 8)
    ax4.set_ylabel("Accuracy")
    src_name = "CV" if "cv_preds" in results else "Test"
    ax4.set_title(f"Per-Class Accuracy ({src_name})", fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.set_ylim(0, 1.15)
    ax4.grid(axis="y", alpha=0.25)

    # Nonzero summary
    nonzero = [(i, float(per_class_acc[i]), int(counts[i])) for i in range(n_cls) if present[i] and not np.isnan(per_class_acc[i]) and per_class_acc[i] > 0]
    if nonzero:
        nonzero.sort(key=lambda x: (-x[1], -x[2]))
        top = nonzero[:10]
        lines = ["Nonzero classes (top 10):", *[f"{idx_to_label.get(i, i)}: {acc:.2f} ({cnt})" for i, acc, cnt in top]]
        ax4.text(0.99, 0.55, "\n".join(lines), transform=ax4.transAxes, ha="right", va="top", fontsize=7, color="#333333", bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", alpha=0.7))

    # Panel 5
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#FFFFFF")
    ax5.axis("off")
    lines = [
        ("Model", FULL_NAMES[key].replace("\n", " ")),
        ("", ""),
        ("K-Fold (K=4)", ""),
        ("  CV Mean",   f"{results['cv_mean']:.4f}"),
        ("  CV Std",    f"± {results['cv_std']:.4f}"),
        ("  Best Fold", f"{max(fold_accs):.4f}" if fold_accs else "—"),
        ("  Worst Fold",f"{min(fold_accs):.4f}" if fold_accs else "—"),
        ("", ""),
        ("Final Results", ""),
        ("  Test Acc",  f"{results['test_acc']:.4f}"),
        ("  Test N",    f"{len(all_labels)}"),
        ("  Classes",   f"{n_cls}"),
        ("", ""),
        ("Published Results", ""),
        ("  Test Acc",  f"{pub['test']:.3f}" if pub['test'] else "—"),
        ("  CV Mean",   f"{pub['cv']:.3f}"   if pub['cv']   else "—"),
    ]
    y_pos = 0.97
    for label, val in lines:
        if label == "" and val == "":
            y_pos -= 0.025
            continue
        is_header = (val == "" and label != "")
        weight = "bold" if is_header else "normal"
        size = 10 if is_header else 9
        ax5.text(0.02, y_pos, label, transform=ax5.transAxes, fontsize=size, fontweight=weight, va="top", color="#333333" if not is_header else "#000000")
        if val:
            ax5.text(0.62, y_pos, val, transform=ax5.transAxes, fontsize=9, va="top", color="#000000")
        y_pos -= 0.055

    fig.suptitle(f"Model Results: {FULL_NAMES[key].replace(chr(10), ' ')}", fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved chart → {save_path}")

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
    if not results:
        print("No JSON results found. Please run the training scripts first.")
        sys.exit(0)

    # Generate per-model charts using same keys/naming as main.py
    for key, res in results.items():
        chart_path = os.path.join(CHARTS_DIR, f"{key.replace('-','_')}_results.png")
        plot_model_results(key, res, chart_path)

    # Comparison overview (if more than one model available)
    if len(results) > 1:
        overview_path = os.path.join(CHARTS_DIR, "comparison_overview.png")
        plot_comparison_overview(results, overview_path)

    # Optional: save confusion matrices and histories as before
    plot_confusion_matrices(results)
    plot_histories(results)
