import os
import sys
import json
import shutil
import argparse
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────
ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JSON_PATH   = os.path.join(ROOT, "dataset", "WLASL_v0.3.json")
VIDEOS_DIR  = os.path.join(ROOT, "dataset", "videos")
DATASET_DIR = os.path.join(ROOT, "dataset")


def organize(subset=None, split=None, copy=True):

    # ── Load JSON ─────────────────────────────────────────────────────────────
    if not os.path.exists(JSON_PATH):
        print(f"❌ JSON not found: {JSON_PATH}")
        print("   Make sure WLASL_v0.3.json is inside your data/ folder.")
        sys.exit(1)

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    print(f"✅ Loaded JSON: {len(data)} glosses total")

    # ── Apply subset (top-K glosses) ──────────────────────────────────────────
    if subset:
        data = data[:subset]
        print(f"   Using subset: top {subset} glosses (WLASL{subset})")

    # ── Stats tracking ────────────────────────────────────────────────────────
    copied    = 0
    skipped   = 0   # video file not found in data/videos/
    filtered  = 0   # excluded by split filter

    os.makedirs(DATASET_DIR, exist_ok=True)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for entry in tqdm(data, desc="Organizing glosses"):
        gloss     = entry["gloss"]          # e.g. "hello"
        instances = entry["instances"]      # list of video instances

        gloss_dir = os.path.join(DATASET_DIR, gloss)
        os.makedirs(gloss_dir, exist_ok=True)

        for instance in instances:
            video_id    = str(instance["video_id"])
            inst_split  = instance.get("split", "train")

            # ── Split filter ──────────────────────────────────────────────────
            if split and inst_split != split:
                filtered += 1
                continue

            # ── Find source video ─────────────────────────────────────────────
            src = os.path.join(VIDEOS_DIR, video_id + ".mp4")

            if not os.path.exists(src):
                skipped += 1
                continue

            # ── Destination ───────────────────────────────────────────────────
            dst = os.path.join(gloss_dir, video_id + ".mp4")

            if os.path.exists(dst):
                continue   # already organized, skip

            if copy:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)

            copied += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    action = "Copied" if copy else "Moved"

    print(f"\n{'='*50}")
    print(f"  {action}   : {copied} videos")
    print(f"  Skipped  : {skipped} videos (file not found in data/videos/)")
    print(f"  Filtered : {filtered} videos (split filter applied)")
    print(f"{'='*50}")
    print(f"\n✅ Dataset organized → {DATASET_DIR}")

    # ── Show class distribution ───────────────────────────────────────────────
    classes = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]
    print(f"\n   Total classes : {len(classes)}")

    # Show classes with video count
    class_counts = {}
    for c in classes:
        count = len([
            f for f in os.listdir(os.path.join(DATASET_DIR, c))
            if f.endswith(".mp4")
        ])
        class_counts[c] = count

    # Warn about classes with very few videos
    low = {k: v for k, v in class_counts.items() if v < 3}
    if low:
        print(f"\n⚠️  Classes with fewer than 3 videos (may hurt training):")
        for k, v in sorted(low.items(), key=lambda x: x[1]):
            print(f"     {k}: {v} video(s)")

    total_videos = sum(class_counts.values())
    print(f"\n   Total videos  : {total_videos}")
    print(f"   Avg per class : {total_videos / max(len(classes), 1):.1f}")

    # Save a summary JSON for reference
    summary = {
        "total_classes": len(classes),
        "total_videos" : total_videos,
        "class_counts" : class_counts
    }
    summary_path = os.path.join(ROOT, "data", "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n   Summary saved → {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize WLASL flat videos into class subfolders"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Use only the top-K glosses. E.g. --subset 100 for WLASL100"
    )
    parser.add_argument(
        "--split", type=str, default=None,
        choices=["train", "val", "test"],
        help="Only include videos from a specific split"
    )
    parser.add_argument(
        "--move", action="store_true",
        help="Move files instead of copying (faster but destructive)"
    )

    args = parser.parse_args()

    organize(
        subset = args.subset,
        split  = args.split,
        copy   = not args.move
    )