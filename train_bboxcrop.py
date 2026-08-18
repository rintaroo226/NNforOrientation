"""bbox クロップ前処理を使った学習スクリプト (train.py のbboxクロップ版)。

train.py が使う SilhouettePoseDataset は画像全体をそのままリサイズするため、
対象までの距離が学習時(distance=10)と変わると精度が大きく落ちることが
eval_distance_sweep.py で確認された。このスクリプトは bboxcrop_dataset.py の
SilhouettePoseBBoxCropDataset (前景bboxを一定の余白付きで切り出してから
正方形にリサイズする)を使って学習し、距離が変わっても姿勢推定できる
モデルを目指す。

学習データ自体は distance=10 で描画された既存のもの(例: matlab/database_random)
をそのまま使う。bboxクロップ後の見え方は元の距離に依存しなくなるため、
複数距離で再レンダリングし直す必要は無い、という想定に基づく。

解像度劣化augmentationは、「綺麗な版」と「劣化させた版」を別々のサンプルとして
データセットに追加する(ConcatDataset、学習データは実質2倍になる)。同じ元姿勢の
綺麗版/劣化版が学習用・検証用に分かれてリークしないよう、姿勢のインデックス単位で
先に train/val を分割してから両方に適用する。検証データは常に綺麗な版のみを使う
(augmentationをかけるとbest_angleの比較がepochごとにブレるため)。

1epochあたりの処理量(=データセットのサイズそのもの、batch_sizeには依らない)は
augmentation有効時に2倍になるため、総計算量(≒学習時間、epoch数×学習データ枚数)を
augmentation無しのベースラインと揃えたい場合は --epochs をベースラインの半分にする
(batch_sizeを変えても総計算量は変わらないので、そちらでは調整できない)。

train.py は変更せず、このスクリプトとして独立に用意する。チェックポイントも
既定で train.py とは別名(checkpoints/silhouette_pose_bboxcrop.pt)に保存する。

使用例:
    python train_bboxcrop.py \
        --data-root matlab/database_random \
        --labels-csv matlab/database_random/labels.csv
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from bboxcrop_dataset import SilhouettePoseBBoxCropDataset
from silhouette_pose.losses import (
    symmetry_aware_angle_error_deg,
    symmetry_aware_quaternion_loss,
)
from silhouette_pose.model import SilhouettePoseNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--output", default="checkpoints/silhouette_pose_bboxcrop.pt")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--padding-factor", type=float, default=1.25,
                         help="前景bboxの外側に取る余白の倍率 (距離によらず一定にする正規化)")
    parser.add_argument("--no-augment-resolution", dest="augment_resolution",
                         action="store_false",
                         help="解像度劣化augmentation(遠距離のガタガタしたエッジを模擬した"
                              "サンプルを追加すること)を無効化する")
    parser.add_argument("--augment-min-scale", type=float, default=0.15,
                         help="解像度劣化augmentationで許容する最小の縮小率")
    parser.set_defaults(augment_resolution=True)
    parser.add_argument("--epochs", type=int, default=50,
                         help="augmentation有効時はデータが2倍になるため、他条件と"
                              "総計算量(≒学習時間)を揃えたい場合はこの値を半分にする")
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

    clean_dataset = SilhouettePoseBBoxCropDataset(
        root=args.data_root,
        labels_csv=args.labels_csv,
        image_size=args.image_size,
        padding_factor=args.padding_factor,
        augment_resolution=False,
    )
    n_samples = len(clean_dataset)
    val_size = max(1, int(n_samples * args.val_ratio))

    # 姿勢インデックス単位で先にtrain/valを分割する(綺麗版/劣化版が
    # 別々に学習用・検証用へ漏れてリークするのを防ぐため)
    perm = torch.randperm(n_samples, generator=torch.Generator().manual_seed(args.seed)).tolist()
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    if args.augment_resolution:
        degraded_dataset = SilhouettePoseBBoxCropDataset(
            root=args.data_root,
            labels_csv=args.labels_csv,
            image_size=args.image_size,
            padding_factor=args.padding_factor,
            augment_resolution=True,
            augment_min_scale=args.augment_min_scale,
        )
        train_set = ConcatDataset([
            Subset(clean_dataset, train_indices),
            Subset(degraded_dataset, train_indices),
        ])
        # 検証データは常に綺麗な版のみ(augmentationをかけるとepochごとに
        # best_angleの比較がブレるため)
        val_set = Subset(clean_dataset, val_indices)
    else:
        train_set = Subset(clean_dataset, train_indices)
        val_set = Subset(clean_dataset, val_indices)

    # 総計算量の目安 = epoch数 x 1epochあたりの学習サンプル数。batch_sizeは
    # 総計算量に影響しない(batchの切り方が変わるだけ)ので調整しない。
    # augmentation有りだとデータが2倍になるため、ベースラインと計算時間を
    # 揃えたいなら --epochs をベースラインの半分にする。
    total_sample_passes = args.epochs * len(train_set)
    print(f"学習データ: {len(train_set)}枚  検証データ: {len(val_set)}枚  "
          f"batch_size: {args.batch_size}")
    print(f"総計算量の目安(epoch数 x 学習データ枚数): {total_sample_passes:,}  "
          f"(他条件と計算時間を揃えたい場合はこの値が一致するようepoch数を調整してください)")

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
                    "augment_resolution": args.augment_resolution,
                    "augment_min_scale": args.augment_min_scale,
                    "best_angle_deg": best_angle,
                },
                output,
            )

    print(f"saved best checkpoint to {output}")


if __name__ == "__main__":
    main()
