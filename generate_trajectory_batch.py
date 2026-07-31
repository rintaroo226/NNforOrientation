"""複数本の剛体回転軌道をまとめて生成する (torch/MATLAB不要)。

generate_trajectory.py は1本ずつ軌道を作るが、統計的な検証には複数本の
軌道 (初期姿勢・初期角速度がそれぞれ独立にランダム) が要る。このスクリプトは
そのループを行い、matlab/render_trajectory_batch.m が読み込める形で
`--out-dir` 以下に traj_000.csv, traj_001.csv, ... を書き出す。

実行例:
    python3 generate_trajectory_batch.py --out-dir trajectories --n-trajectories 10
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from quat_np import sample_uniform_quat
from rigid_body import integrate_trajectory, principal_inertia


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-trajectories", type=int, default=10)
    p.add_argument("--duration", type=float, default=15.0, help="軌道1本の長さ [秒]")
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--box-size", default="3,2,1")
    p.add_argument("--process-model", choices=["constant", "euler"], default="euler",
                    help="箱を放り投げた状況を再現するので既定は euler")
    p.add_argument("--omega-mag-range", default="1.0,4.0",
                    help="初期角速度の大きさをこの範囲 [rad/s] から一様サンプリングする")
    p.add_argument("--seed", type=int, default=100, help="軌道iは seed+i を使う (再現性のため)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    box_size = tuple(float(v) for v in args.box_size.split(","))
    inertia = principal_inertia(box_size)
    omega_mag_lo, omega_mag_hi = (float(v) for v in args.omega_mag_range.split(","))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / args.fps
    n_steps = int(round(args.duration * args.fps))

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["traj_id", "csv", "seed", "q0", "omega0"])

        for i in range(args.n_trajectories):
            seed = args.seed + i
            rng = np.random.default_rng(seed)

            q0 = sample_uniform_quat(rng).astype(np.float64)
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            mag = rng.uniform(omega_mag_lo, omega_mag_hi)
            omega0 = axis * mag

            times, quats, omegas = integrate_trajectory(
                q0, omega0, dt, n_steps, inertia, args.process_model
            )

            traj_name = f"traj_{i:03d}"
            csv_path = out_dir / f"{traj_name}.csv"
            with csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "qw", "qx", "qy", "qz", "wx", "wy", "wz"])
                for k in range(len(times)):
                    w.writerow([
                        f"{times[k]:.6f}",
                        *[f"{v:.8f}" for v in quats[k]],
                        *[f"{v:.8f}" for v in omegas[k]],
                    ])

            writer.writerow([traj_name, csv_path.name, seed,
                              " ".join(f"{v:.6f}" for v in q0),
                              " ".join(f"{v:.6f}" for v in omega0)])
            print(f"[{i+1}/{args.n_trajectories}] {traj_name}: "
                  f"|omega0|={mag:.2f} rad/s, 軸={axis.round(2)} -> {csv_path}")

    print(f"\n完了。{args.n_trajectories}本の軌道を {out_dir}/ に保存しました "
          f"(manifest: {manifest_path})。")


if __name__ == "__main__":
    main()
