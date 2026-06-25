import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SilhouettePoseDataset(Dataset):
    """Load silhouette images and quaternion labels from a CSV file.

    CSV format:
        image,qw,qx,qy,qz
        silhouettes/sample_000001.png,0.98,0.01,0.02,0.19
    """

    def __init__(
        self,
        root: str | Path,
        labels_csv: str | Path,
        image_size: int = 64,
        threshold: float | None = 0.5,
    ) -> None:
        self.root = Path(root)
        self.labels_csv = Path(labels_csv)
        self.image_size = image_size
        self.threshold = threshold
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
        image = Image.open(image_path).convert("L").resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if self.threshold is not None:
            arr = (arr >= self.threshold).astype(np.float32)
        x = torch.from_numpy(arr).unsqueeze(0)
        y = torch.from_numpy(quat)
        return x, y
