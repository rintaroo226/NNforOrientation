"""bboxクロップ前処理の各段階を、複数の距離×複数の姿勢で並べて可視化する。

eval_distance_sweep_bboxcrop.py で遠距離ほど精度が悪化した原因が
bboxクロップの実装バグなのか、元のレンダリング解像度(512x512固定)が
遠距離で箱を粗くしか描けていないためなのかを目視で確認する。

各サンプルについて次の3段階を並べて表示する(実際にモデルに入る画像は
一番右の列そのもの):
  1. 元のレンダリング画像(512x512) + bbox(赤)/クロップ領域(黄)の重畳
  2. 元画像からそのままクロップした領域(リサイズ前、実寸で表示。
     ここでの見た目の粗さがそのまま情報量の上限になる)
  3. image_size(既定64x64)にリサイズ後 — bboxcrop_dataset.bbox_crop_resize()
     の出力そのもので、モデルへの実際の入力

同じ姿勢セットが全ての距離で使い回されている
(matlab/generate_distance_sweep_database.m の設計)ことを利用し、
同一姿勢を複数距離で比較できるようにする。torch非依存 (numpy/PIL/matplotlibのみ)。

使用例:
    python3 visualize_bboxcrop_preprocessing.py \
        --data-root matlab/database_distance_sweep \
        --labels-csv matlab/database_distance_sweep/labels.csv \
        --distances 10,40,55 --n-poses 3
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from bboxcrop_dataset import bbox_crop_resize, tight_bbox


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--labels-csv", required=True,
                    help="image,qw,qx,qy,qz,distance 形式のCSV "
                         "(matlab/generate_distance_sweep_database.m の出力)")
    p.add_argument("--out", default="bboxcrop_preprocessing_visualization.png")
    p.add_argument("--padding-factor", type=float, default=1.25,
                    help="train_bboxcrop.py の --padding-factor と合わせる")
    p.add_argument("--image-size", type=int, default=64,
                    help="モデルへの実際の入力サイズ")
    p.add_argument("--distances", default="10,40,55",
                    help="可視化する距離をカンマ区切りで指定")
    p.add_argument("--n-poses", type=int, default=3,
                    help="各距離について、姿勢セット内から等間隔で抽出する数")
    return p.parse_args()


def load_rows_grouped_by_distance(path: Path) -> dict[float, list[str]]:
    """距離ごとに、labels.csv内での出現順を保ったまま画像パスのリストを返す。"""
    grouped: dict[float, list[str]] = defaultdict(list)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[float(row["distance"])].append(row["image"])
    return grouped


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    target_distances = [float(v) for v in args.distances.split(",")]
    grouped = load_rows_grouped_by_distance(Path(args.labels_csv))

    for d in target_distances:
        if d not in grouped:
            raise ValueError(f"distance={d} が labels.csv 内に見つかりません "
                              f"(利用可能: {sorted(grouped.keys())})")

    n_poses = args.n_poses
    sample_count = len(grouped[target_distances[0]])
    pose_indices = np.linspace(0, sample_count - 1, n_poses).astype(int)

    rows = [(d, idx) for d in target_distances for idx in pose_indices]
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(9.5, 3.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for r, (d, pose_idx) in enumerate(rows):
        image_path = root / grouped[d][pose_idx]
        arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        x0, y0, x1, y1 = tight_bbox(arr)
        bbox_size = max(x1 - x0, y1 - y0)
        crop_size = bbox_size * args.padding_factor
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

        # ① 元画像 + bbox/クロップ領域の重畳
        ax0 = axes[r, 0]
        ax0.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax0.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, edgecolor="red", facecolor="none", linewidth=1.2))
        ax0.add_patch(patches.Rectangle(
            (cx - crop_size / 2, cy - crop_size / 2), crop_size, crop_size,
            edgecolor="yellow", facecolor="none", linewidth=1.2, linestyle="--"))
        ax0.set_title(f"d={d:.0f} pose#{pose_idx}\n元画像 bbox={bbox_size}px", fontsize=8)
        ax0.axis("off")

        # ② リサイズ前のクロップ領域を実寸(等倍)で表示 → 実際の情報量の粗さが見える
        half = crop_size / 2.0
        h, w = arr.shape
        left, top = cx - half, cy - half
        right, bottom = cx + half, cy + half
        pad_left = max(0, int(np.ceil(-left)))
        pad_top = max(0, int(np.ceil(-top)))
        pad_right = max(0, int(np.ceil(right - w)))
        pad_bottom = max(0, int(np.ceil(bottom - h)))
        padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
        left_p = int(round(left + pad_left))
        top_p = int(round(top + pad_top))
        size_p = int(round(crop_size))
        raw_crop = padded[top_p:top_p + size_p, left_p:left_p + size_p]

        ax1 = axes[r, 1]
        ax1.imshow(raw_crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax1.set_title(f"クロップ領域(等倍) {raw_crop.shape[1]}x{raw_crop.shape[0]}px", fontsize=8)
        ax1.axis("off")

        # ③ 最終的にモデルへ入力される画像 (bbox_crop_resize の出力そのもの)
        model_input = bbox_crop_resize(arr, args.image_size, args.padding_factor)
        ax2 = axes[r, 2]
        ax2.imshow(model_input, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax2.set_title(f"モデル入力 {args.image_size}x{args.image_size}", fontsize=8)
        ax2.axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
