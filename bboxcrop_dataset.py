"""bbox クロップ前処理(距離不変な正規化)の中核となる純粋関数群。

対象までの距離が変わると画面内での見かけの大きさが変わり、姿勢推定精度が
大きく落ちることが eval_distance_sweep.py で確認された。ここでは前景の
外接矩形(bbox)を一定の余白付きで切り出してから正方形にリサイズすることで、
「対象がフレームに占める割合」を常に一定にする(距離不変な正規化)。

この処理は決定論的(degrade_resolutionのみランダム性あり)なので、学習の
Dataset の __getitem__ 内で毎epochやり直すのではなく、
preprocess_bboxcrop_dataset.py でオフライン(学習前に1回だけ)に適用し、
結果を新しい画像ファイルとして保存する方式にしている。学習側は生成済みの
画像を silhouette_pose.dataset.SilhouettePoseDataset でそのまま読むだけでよく、
このモジュールに依存しない。

eval_distance_sweep_bboxcrop.py 等、推論時に(保存済みでない)新しい画像へ
その場でクロップを適用する必要がある箇所からはこの関数群を直接importする。
"""
import numpy as np
from PIL import Image


def tight_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    """2値シルエット画像(0/255)から前景の外接矩形 (x0, y0, x1, y1) を求める。"""
    ys, xs = np.nonzero(arr > 0)
    h, w = arr.shape
    if len(xs) == 0:
        return 0, 0, w, h
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def bbox_crop_resize(arr: np.ndarray, image_size: int, padding_factor: float) -> np.ndarray:
    """前景bboxの周りに padding_factor 倍の正方形領域を取り、image_size にリサイズする。

    対象までの距離が変わって見かけの大きさが変わっても、この処理を通せば
    出力画像内で対象が占める割合は常に一定になる(距離不変な正規化)。
    フレーム外にはみ出す場合はゼロ埋め(黒)でパディングする。
    """
    x0, y0, x1, y1 = tight_bbox(arr)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bbox_size = max(x1 - x0, y1 - y0)
    crop_size = max(bbox_size * padding_factor, 1.0)
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
    cropped = padded[top_p:top_p + size_p, left_p:left_p + size_p]

    img = Image.fromarray(cropped).resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def degrade_resolution(arr01: np.ndarray, image_size: int, min_scale: float = 0.15) -> np.ndarray:
    """image_size の画像を一度ランダムに小さい解像度へダウンサンプルしてから
    image_size に戻す (実効解像度がガタガタ/低解像度になる状況を模擬する)。

    distance=10 固定で学習すると、bboxクロップ後も常に高い実効解像度の画像しか
    見ないため、遠距離レンダリングで生じるガタガタしたエッジにNNが慣れておらず
    精度が落ちることが distance sweep 評価と可視化で確認された。学習時にこの
    augmentation を掛けることで、様々な実効解像度に対して頑健にする狙い。
    """
    scale = np.random.uniform(min_scale, 1.0)
    small_size = max(int(round(image_size * scale)), 4)
    img = Image.fromarray((arr01 * 255.0).astype(np.uint8))
    img = img.resize((small_size, small_size), Image.Resampling.BILINEAR)
    img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0
