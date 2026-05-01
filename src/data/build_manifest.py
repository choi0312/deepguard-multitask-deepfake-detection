from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

METHOD_MAP: Dict[str, int] = {
    "Deepfakes": 0,
    "FaceSwap": 1,
    "Face2Face": 2,
    "NeuralTextures": 3,
}


def infer_video_id(path: Path) -> str:
    """Infer video id from image filename."""
    stem = path.stem
    return stem.split("_frame")[0].split("_")[0]


def collect_rows(data_root: Path) -> List[dict]:
    rows: List[dict] = []

    for class_dir in sorted(data_root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        if class_name == "Real":
            label_rf = 0
            label_method = -1
        elif class_name in METHOD_MAP:
            label_rf = 1
            label_method = METHOD_MAP[class_name]
        else:
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            rows.append(
                {
                    "image_path": str(image_path),
                    "class_name": class_name,
                    "video_id": infer_video_id(image_path),
                    "label_rf": label_rf,
                    "label_method": label_method,
                }
            )

    if not rows:
        raise RuntimeError(f"No image files found under {data_root}")

    return rows


def split_dataframe(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_by: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)

    if split_by == "video_id":
        keys = sorted(df["video_id"].unique().tolist())
        rng.shuffle(keys)

        n_total = len(keys)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)

        test_keys = set(keys[:n_test])
        val_keys = set(keys[n_test : n_test + n_val])

        test_df = df[df["video_id"].isin(test_keys)]
        val_df = df[df["video_id"].isin(val_keys)]
        train_df = df[~df["video_id"].isin(test_keys | val_keys)]
    else:
        indices = list(df.index)
        rng.shuffle(indices)

        n_total = len(indices)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)

        test_idx = set(indices[:n_test])
        val_idx = set(indices[n_test : n_test + n_val])

        test_df = df.loc[list(test_idx)]
        val_df = df.loc[list(val_idx)]
        train_df = df.drop(index=list(test_idx | val_idx))

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--split_by", type=str, choices=["video_id", "image"], default="video_id")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(data_root)
    df = pd.DataFrame(rows)

    train_df, val_df, test_df = split_dataframe(
        df=df,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_by=args.split_by,
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    label_map = {
        "real": 0,
        "fake": 1,
        "methods": METHOD_MAP,
    }

    with (output_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print("Manifest files created:")
    print("train:", len(train_df))
    print("val:", len(val_df))
    print("test:", len(test_df))


if __name__ == "__main__":
    main()
