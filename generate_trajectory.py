"""剛体回転の連続軌道 (クォータニオン + 角速度の時系列) を生成する (torch/MATLAB不要)。

EKF のオフライン検証用に、既知の運動方程式に従う「正解」軌道を作る。
出力される CSV (t,qw,qx,qy,qz,wx,wy,wz) は、matlab/render_trajectory.m に渡して
画像シーケンスをレンダリングするための入力になる。

実行例:
    python3 generate_trajectory.py --out-csv trajectory.csv \
        --duration 5 --fps 60 --process-model euler --omega0 0.1,3.0,0.2
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from quat_np import sample_uniform_quat
from rigid_body import integrate_trajectory, principal_inertia


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-csv", required=True)
    p.add_argument("--duration", type=float, default=5.0, help="軌道の長さ [秒]")
    p.add_argument("--fps", type=float, default=60.0, help="フレームレート [Hz]")
    p.add_argument("--box-size", default="3,2,1",
                    help="直方体サイズ 幅,高さ,奥行 [m] (generate_random_database.m の target_size と合わせる)")
    p.add_argument("--process-model", choices=["constant", "euler"], default="euler",
                    help="運動モデル (constant: ω一定, euler: 無トルク剛体の運動方程式)")
    p.add_argument("--omega0", default=None,
                    help="初期角速度 wx,wy,wz [rad/s] (未指定ならランダム。任意方向を許容する)")
    p.add_argument("--omega0-mag", type=float, default=3.0,
                    help="--omega0 未指定時のランダム角速度の大きさ [rad/s]")
    p.add_argument("--q0", default=None,
                    help="初期姿勢 qw,qx,qy,qz (未指定なら SO(3) 上一様サンプリング)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    box_size = tuple(float(v) for v in args.box_size.split(","))
    inertia = principal_inertia(box_size)

    if args.q0 is not None:
        q0 = np.array([float(v) for v in args.q0.split(",")])
    else:
        q0 = sample_uniform_quat(rng).astype(np.float64)

    if args.omega0 is not None:
        omega0 = np.array([float(v) for v in args.omega0.split(",")])
    else:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        omega0 = axis * args.omega0_mag

    dt = 1.0 / args.fps
    n_steps = int(round(args.duration * args.fps))

    times, quats, omegas = integrate_trajectory(q0, omega0, dt, n_steps, inertia, args.process_model)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "qw", "qx", "qy", "qz", "wx", "wy", "wz"])
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.6f}",
                *[f"{v:.8f}" for v in quats[i]],
                *[f"{v:.8f}" for v in omegas[i]],
            ])

    print(f"box_size={box_size}, inertia_ratio={inertia}")
    print(f"process_model={args.process_model}, q0={q0}, omega0={omega0}")
    print(f"{len(times)}フレーム ({args.duration}秒 @ {args.fps}fps) を書き出しました: {out_path}")


if __name__ == "__main__":
    main()
