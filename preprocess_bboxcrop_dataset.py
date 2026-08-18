"""既存のシルエット画像データセットに bbox クロップ(+解像度劣化augmentation)を
オフラインで一括適用し、新しい画像ファイル一式として書き出す前処理スクリプト。

以前は train_bboxcrop.py の Dataset の __getitem__ 内でこの処理(bbox検出+
クロップ+リサイズ)を毎epoch・毎サンプルやり直していたが、これは決定論的な
処理(劣化augmentationの強さだけがランダム)なので、学習ループの中で何度も
繰り返すのは無駄な計算コストになる。ここで1回だけ処理して新しい画像ファイル
として保存しておけば、学習時は普通の SilhouettePoseDataset
(silhouette_pose/dataset.py, 変更無し)でそのまま読み込める。

--augment-resolution を指定すると、各画像について「綺麗な版」に加えて
「劣化させた版」も別サンプルとして書き出す(出力の学習データは実質2倍になる)。

使用例:
    python3 preprocess_bboxcrop_dataset.py \
        --data-root matlab/database_random \
        --labels-csv matlab/database_random/labels.csv \
        --out-dir database_random_bboxcrop \
        --augment-resolution
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

from bboxcrop_dataset import bbox_crop_resize, degrade_resolution


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--labels-csv", required=True,
                    help="image,qw,qx,qy,qz 形式のCSV (前処理前のソースデータセット)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--padding-factor", type=float, default=1.25,
                    help="前景bboxの外側に取る余白の倍率 (距離によらず一定にする正規化)")
    p.add_argument("--augment-resolution", action="store_true",
                    help="各画像について解像度劣化版も追加で書き出す(出力データが2倍になる)")
    p.add_argument("--augment-min-scale", type=float, default=0.15,
                    help="解像度劣化augmentationで許容する最小の縮小率")
    return p.parse_args()


def load_source_labels_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "qw", "qx", "qy", "qz"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"labels CSV is missing columns: {sorted(missing)}")
        for row in reader:
            rows.append({k: row[k] for k in ("image", "qw", "qx", "qy", "qz")})
    return rows


def save_png(arr01: np.ndarray, path: Path) -> None:
    img = Image.fromarray((np.clip(arr01, 0.0, 1.0) * 255.0).astype(np.uint8))
    img.save(path)


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = load_source_labels_csv(Path(args.labels_csv))
    n = len(rows)
    print(f"{n}枚を読み込みました (augmentation: {'有効' if args.augment_resolution else '無効'})")

    out_csv = out_dir / "labels.csv"
    start = time.perf_counter()

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "qw", "qx", "qy", "qz"])
        for i, row in enumerate(rows):
            arr = np.asarray(Image.open(root / row["image"]).convert("L"), dtype=np.uint8)
            cropped = bbox_crop_resize(arr, args.image_size, args.padding_factor)

            rel_clean = f"images/sample_{i:06d}.png"
            save_png(cropped, image_dir / f"sample_{i:06d}.png")
            writer.writerow([rel_clean, row["qw"], row["qx"], row["qy"], row["qz"]])

            if args.augment_resolution:
                degraded = degrade_resolution(cropped, args.image_size, args.augment_min_scale)
                rel_degraded = f"images/sample_{i:06d}_degraded.png"
                save_png(degraded, image_dir / f"sample_{i:06d}_degraded.png")
                writer.writerow([rel_degraded, row["qw"], row["qx"], row["qy"], row["qz"]])

            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{n} 完了 ({time.perf_counter()-start:.1f}秒)")

    elapsed = time.perf_counter() - start
    n_out = n * (2 if args.augment_resolution else 1)
    print(f"\n完了。{n_out}枚を {out_dir}/ に保存しました ({elapsed:.1f}秒)")


if __name__ == "__main__":
    main()
