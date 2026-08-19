"""labels.csv が実際の画像ファイルと正しく対応しているかを検証する。

matlab/rebuild_labels_multidistance.m のように、MATLABの乱数列から
labels.csv を再計算で復元した場合、インデックスのズレなどで画像と
ラベルの対応が壊れていないかを、再レンダリング無しで確認したい。

distance_from_bbox.bbox_px_analytic(q, distance) で「このラベル通りの
姿勢・距離なら、bboxは何ピクセルになるはずか」を幾何学的に計算し、
実際に画像から測ったbboxサイズ(bboxcrop_dataset.tight_bbox)と比較する。
対応が正しければ、両者は数%以内の誤差で一致するはず(distance_from_bbox.py
で database_distance_sweep を使って事前に検証済みの精度と同水準)。
大きくズレるサンプルが多ければ、ラベルと画像の対応がズレている疑いが強い。

torch不要 (numpy, PIL のみ)。

使用例:
    python3 verify_recovered_labels.py \
        --data-root matlab/database_random_multidistance_10 \
        --labels-csv matlab/database_random_multidistance_10/labels.csv \
        --n-samples 200
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from bboxcrop_dataset import tight_bbox
from distance_from_bbox import bbox_px_analytic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--box-size", default="3,2,1")
    p.add_argument("--n-samples", type=int, default=200,
                    help="全行から等間隔で抽出して検証する枚数(全部やると遅いため)")
    p.add_argument("--err-threshold-pct", type=float, default=5.0,
                    help="この誤差%を超えたら「対応がズレている疑い」として個別に表示する")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    box_size = tuple(float(v) for v in args.box_size.split(","))
    root = Path(args.data_root)

    with Path(args.labels_csv).open(newline="") as f:
        rows = list(csv.DictReader(f))
    n_total = len(rows)
    print(f"labels.csv: {n_total}行")

    rng = np.random.default_rng(args.seed)
    n_sample = min(args.n_samples, n_total)
    indices = rng.choice(n_total, size=n_sample, replace=False)
    indices.sort()

    errs = []
    suspicious = []
    n_no_bbox = 0
    n_clipped = 0
    for idx in indices:
        row = rows[idx]
        q = np.array([float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])])
        q /= max(np.linalg.norm(q), 1e-8)
        distance = float(row["distance"])

        image_path = root / row["image"]
        arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        x0, y0, x1, y1 = tight_bbox(arr)
        observed_bbox_px = max(x1 - x0, y1 - y0)
        if observed_bbox_px <= 0:
            n_no_bbox += 1
            continue
        h, w = arr.shape
        touches_border = (x0 == 0) or (x1 == w) or (y0 == 0) or (y1 == h)
        if touches_border:
            # 箱がフレーム端で切れている(クリッピング)場合、実測bboxは理論値より
            # 小さくなるのが正常なので、対応がズレているかの判定からは除外する
            # (distance_from_bbox.py の事前検証でも同じ理由で除外している)。
            n_clipped += 1
            continue

        expected_bbox_px = bbox_px_analytic(q, distance, box_size)
        err_pct = abs(expected_bbox_px - observed_bbox_px) / observed_bbox_px * 100
        errs.append(err_pct)
        if err_pct > args.err_threshold_pct:
            suspicious.append((row["image"], err_pct, expected_bbox_px, observed_bbox_px))

    errs = np.array(errs)
    print(f"\n検証したサンプル数: {len(errs)} "
          f"(前景ピクセル無し: {n_no_bbox}, フレーム端接触のため除外: {n_clipped})")
    print(f"理論bboxサイズ vs 実測bboxサイズ の誤差: "
          f"平均={errs.mean():.2f}%  中央値={np.median(errs):.2f}%  最大={errs.max():.2f}%")
    print(f"誤差{args.err_threshold_pct:.0f}%超のサンプル数: {len(suspicious)}/{len(errs)}")

    if suspicious:
        print(f"\n[疑わしいサンプル(誤差{args.err_threshold_pct:.0f}%超、最大10件)]")
        for image, err_pct, expected, observed in suspicious[:10]:
            print(f"  {image}: 誤差={err_pct:.1f}%  理論値={expected:.1f}px  実測値={observed:.1f}px")

    if np.median(errs) < 3.0 and len(suspicious) / max(len(errs), 1) < 0.05:
        print("\n判定: ラベルと画像の対応は正常と考えられます"
              "(distance_from_bbox.py の事前検証と同水準の誤差)。")
    else:
        print("\n判定: 誤差が事前検証時より明らかに大きい、または疑わしいサンプルが多いです。"
              "ラベルと画像の対応がズレている可能性があります。")


if __name__ == "__main__":
    main()
