"""bbox クロップ前処理を使った学習スクリプト (train.py のbboxクロップ版)。

train.py が使う SilhouettePoseDataset は画像全体をそのままリサイズするため、
対象までの距離が学習時(distance=10)と変わると精度が大きく落ちることが
eval_distance_sweep.py で確認された。bboxクロップ(前景の外接矩形を一定の
余白付きで切り出してから正方形にリサイズする)で「対象がフレームに占める
割合」を常に一定にすれば、距離が変わっても姿勢推定できるはず、という狙い。

bboxクロップ自体(+解像度劣化augmentation)は決定論的な前処理なので、この
スクリプトの中では行わない。事前に preprocess_bboxcrop_dataset.py で1回だけ
適用し、新しい画像ファイル一式として保存しておいたものを、普通の
SilhouettePoseDataset (silhouette_pose/dataset.py, 変更無し)でそのまま
読み込む。学習コードは train.py とほぼ同じままにして、前処理は前処理側の
責務として分離する。

train.py は変更せず、このスクリプトとして独立に用意する。チェックポイントも
既定で train.py とは別名(checkpoints/silhouette_pose_bboxcrop.pt)に保存する。

使用例:
    python3 preprocess_bboxcrop_dataset.py \
        --data-root matlab/database_random \
        --labels-csv matlab/database_random/labels.csv \
        --out-dir database_random_bboxcrop \
        --augment-resolution
    python train_bboxcrop.py \
        --data-root database_random_bboxcrop \
        --labels-csv database_random_bboxcrop/labels.csv
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from silhouette_pose.dataset import SilhouettePoseDataset
from silhouette_pose.losses import (
    symmetry_aware_angle_error_deg,
    symmetry_aware_quaternion_loss,
)
from silhouette_pose.model import SilhouettePoseNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True,
                         help="preprocess_bboxcrop_dataset.py の --out-dir")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--output", default="checkpoints/silhouette_pose_bboxcrop.pt")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--padding-factor", type=float, default=1.25,
                         help="preprocess_bboxcrop_dataset.py で使った値と揃える"
                              "(学習自体には使わず、チェックポイントに記録するだけ。"
                              "推論時に同じクロップを再現するため)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    model: SilhouettePoseNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_angle = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = symmetry_aware_quaternion_loss(pred, y)
            angle = symmetry_aware_angle_error_deg(pred, y)
            batch = x.size(0)
            total_loss += float(loss.item()) * batch
            total_angle += float(angle.mean().item()) * batch
            total_count += batch
    return total_loss / total_count, total_angle / total_count


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    dataset = SilhouettePoseDataset(
        root=args.data_root,
        labels_csv=args.labels_csv,
        image_size=args.image_size,
    )
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"学習データ: {len(train_set)}枚  検証データ: {len(val_set)}枚")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model = SilhouettePoseNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_angle = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    n_train_batches = len(train_loader)
    log_interval = max(1, n_train_batches // 2)  # エポック内2回表示

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = symmetry_aware_quaternion_loss(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * x.size(0)
            train_count += x.size(0)

            if batch_idx % log_interval == 0 or batch_idx == n_train_batches:
                print(
                    f"  epoch={epoch:03d}/{args.epochs}"
                    f"  batch={batch_idx}/{n_train_batches}"
                    f"  loss={loss.item():.4f}",
                    flush=True,
                )

        scheduler.step()
        val_loss, val_angle = evaluate(model, val_loader, device)
        train_loss /= train_count
        print(
            f"epoch={epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_angle_deg={val_angle:.2f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        if val_angle < best_angle:
            best_angle = val_angle
            torch.save(
                {
                    "model": model.state_dict(),
                    "image_size": args.image_size,
                    "padding_factor": args.padding_factor,
                    "best_angle_deg": best_angle,
                },
                output,
            )

    print(f"saved best checkpoint to {output}")


if __name__ == "__main__":
    main()
