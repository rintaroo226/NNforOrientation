"""既存の distance=10 レンダリング画像に bbox クロップ前処理を追加した Dataset。

train.py が使う SilhouettePoseDataset (silhouette_pose/dataset.py) は画像全体を
そのままリサイズするため、対象までの距離が学習時(distance=10)と変わると
画面内での見かけの大きさが変わり、精度が大きく落ちることが
eval_distance_sweep.py で確認された。

この Dataset は、前景の外接矩形(bbox)を一定の余白付きで切り出してから
正方形にリサイズする。この正規化を学習時から使うことで、距離が変わって
対象の見かけの大きさが変わっても「対象がフレームに占める割合」は常に
一定になり、将来 別の距離で撮った画像でも同じ前処理をかければ姿勢推定できる
はず、という考え方に基づく。学習データ自体は distance=10 の既存レンダリング
(matlab/database_random 等)をそのまま使い、複数距離での再レンダリングは行わない。

silhouette_pose/dataset.py は変更せず、この用途専用の新しい Dataset として
独立に実装する。
"""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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


class SilhouettePoseBBoxCropDataset(Dataset):
    """SilhouettePoseDataset(silhouette_pose/dataset.py)のbboxクロップ版。

    CSV format:
        image,qw,qx,qy,qz
        silhouettes/sample_000001.png,0.98,0.01,0.02,0.19
    """

    def __init__(
        self,
        root: str | Path,
        labels_csv: str | Path,
        image_size: int = 64,
        padding_factor: float = 1.25,
    ) -> None:
        self.root = Path(root)
        self.labels_csv = Path(labels_csv)
        self.image_size = image_size
        self.padding_factor = padding_factor
        self.samples = self._read_labels()

    def _read_labels(self) -> list[tuple[str, np.ndarray]]:
        samples: list[tuple[str, np.ndarray]] = []
        with self.labels_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            required = {"image", "qw", "qx", "qy", "qz"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"labels CSV is missing columns: {sorted(missing)}")
            for row in reader:
                quat = np.array(
                    [row["qw"], row["qx"], row["qy"], row["qz"]],
                    dtype=np.float32,
                )
                quat /= max(float(np.linalg.norm(quat)), 1e-8)
                samples.append((row["image"], quat))
        if not samples:
            raise ValueError(f"no samples found in {self.labels_csv}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rel_path, quat = self.samples[index]
        image_path = self.root / rel_path
        arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        cropped = bbox_crop_resize(arr, self.image_size, self.padding_factor)
        x = torch.from_numpy(cropped).unsqueeze(0)
        y = torch.from_numpy(quat)
        return x, y
