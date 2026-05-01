from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocess import iter_images
from src.utils.seed import set_seed

CLASS_TO_INFO: Dict[str, Dict[str, int]] = {
    "Real": {"label_rf": 0, "label_method": -1},
    "Deepfakes": {"label_rf": 1, "label_method": 0},
    "FaceSwap": {"label_rf": 1, "label_method": 1},
    "Face2Face": {"label_rf": 1, "label_method": 2},
    "NeuralTextures": {"label_rf": 1, "label_method": 3},
}


def infer_video_id(image_path: Path) -> str:
    """Infer video ID from frame path.

    Priority:
    1. Parent folder name if frames are grouped per video
    2. Filename prefix before '_frame_'
    3. File stem
    """
    stem = image_path.stem
    if "_frame_" in stem:
        return stem.split("_frame_")[0]
    if image_path.parent.name not in CLASS_TO_INFO:
        return image_path.parent.name
    return stem


def collect_rows(data_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for class_name, info in CLASS_TO_INFO.items():
        class_dir = data_root / class_name
        if not class_dir.exists():
            print(f"[WARN] class directory not found: {class_dir}")
            continue
        for img_path in iter_images(class_dir):
            rows.append({
                "image_path": str(img_path),
                "video_id": infer_video_id(img_path),
                "label_rf": info["label_rf"],
                "label_method": info["label_method"],
                "method_name": class_name,
            })
    if not rows:
        raise RuntimeError(f"No images found under {data_root}")
    return pd.DataFrame(rows)


def split_by_video(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> pd.DataFrame:
    video_df = df[["video_id", "label_rf", "label_method", "method_name"]].drop_duplicates("video_id")
    stratify_col = video_df["method_name"] if video_df["method_name"].value_counts().min() >= 2 else None

    train_video, temp_video = train_test_split(
        video_df,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=stratify_col,
    )
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    stratify_temp = temp_video["method_name"] if temp_video["method_name"].value_counts().min() >= 2 else None
    val_video, test_video = train_test_split(
        temp_video,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=stratify_temp,
    )

    split_map = {vid: "train" for vid in train_video["video_id"]}
    split_map.update({vid: "val" for vid in val_video["video_id"]})
    split_map.update({vid: "test" for vid in test_video["video_id"]})

    df = df.copy()
    df["split"] = df["video_id"].map(split_map)
    return df


def split_by_frame(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> pd.DataFrame:
    stratify = df["method_name"] if df["method_name"].value_counts().min() >= 2 else None
    train_df, temp_df = train_test_split(df, test_size=val_ratio + test_ratio, random_state=seed, stratify=stratify)
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    stratify_temp = temp_df["method_name"] if temp_df["method_name"].value_counts().min() >= 2 else None
    val_df, test_df = train_test_split(temp_df, test_size=relative_test_ratio, random_state=seed, stratify=stratify_temp)
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test manifests from class folders.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/manifests")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--split_by", type=str, choices=["video_id", "frame"], default="video_id")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_rows(data_root)
    if args.split_by == "video_id":
        df = split_by_video(df, args.val_ratio, args.test_ratio, args.seed)
    else:
        df = split_by_frame(df, args.val_ratio, args.test_ratio, args.seed)

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].reset_index(drop=True)
        split_df.to_csv(output_dir / f"{split}.csv", index=False)
        print(f"[{split}] {len(split_df):,} frames / {split_df['video_id'].nunique():,} videos -> {output_dir / f'{split}.csv'}")

    df.to_csv(output_dir / "all.csv", index=False)
    print(f"[all] {len(df):,} frames -> {output_dir / 'all.csv'}")


if __name__ == "__main__":
    main()
