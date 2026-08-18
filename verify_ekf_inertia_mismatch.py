"""ekf.py の euler プロセスモデルに与える慣性 (inertia) が真値からズレている場合の
感度解析スクリプト (torch/MATLAB/チェックポイント不要)。

背景: EKFConfig(process_model="euler") は箱の主慣性モーメント比を config.inertia
として厳密に知っている前提で設計されている。しかし実運用では質量分布・寸法の
不確かさにより真の慣性は正確にはわからない。ここでは「間違った慣性」を EKF に
与えたとき、ジャニベコフ体制 (中間軸タンブリング) のトラッキング精度がどれだけ
劣化するかを測定する。

実行: /usr/local/bin/python3 verify_ekf_inertia_mismatch.py
"""
import numpy as np

from ekf import BoxOrientationEKF, EKFConfig
from quat_np import quat_from_rotvec, quat_mul, symmetry_aware_angle_error_deg
from rigid_body import integrate_trajectory, principal_inertia

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return condition


def noisy_measurement(q_true: np.ndarray, rng: np.random.Generator, noise_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.normal(0, noise_deg))
    return quat_mul(q_true, quat_from_rotvec(axis * angle))


def run_ekf_with_inertia(
    quats_true: np.ndarray,
    measurements: list[np.ndarray],
    dt: float,
    inertia_used: np.ndarray,
    meas_noise_deg: float,
    omega0_guess: np.ndarray,
) -> np.ndarray:
    """与えられた (誤った可能性のある) inertia_used で EKF を回し、各ステップの
    真値に対する対称性考慮角度誤差 [deg] の配列を返す。"""
    cfg = EKFConfig(
        process_model="euler",
        inertia=inertia_used,
        meas_noise_deg=meas_noise_deg,
        gate_mode="both",
    )
    p0 = np.eye(6)
    p0[:3, :3] *= 0.3
    p0[3:, 3:] *= 0.5
    f = BoxOrientationEKF(quats_true[0], omega0_guess, p0, cfg)
    errs = []
    for k in range(1, len(quats_true)):
        f.predict(dt)
        f.update(measurements[k])
        q_est, _, _ = f.state()
        errs.append(symmetry_aware_angle_error_deg(q_est, quats_true[k]))
    return np.array(errs)


def main() -> None:
    all_passed = True

    # --- セットアップ: ジャニベコフ体制 (中間軸付近のタンブリング) ---
    box_size = (3.0, 2.0, 1.0)
    true_inertia = principal_inertia(box_size)  # (5, 10, 13) 比
    dt = 1.0 / 60.0
    meas_noise_deg = 2.0
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    omega0 = np.array([0.05, 3.0, 0.05])  # 中間軸(y, I=10)付近、verify_ekf.py シナリオDと同一
    n_steps = 900  # 15秒、この間に反転が複数回起きる

    print("=== 感度解析: ekf.py の euler プロセスモデルに与える慣性の誤差 ===\n")
    print(f"真の慣性比 (principal_inertia{box_size}) = {true_inertia}")
    print(f"omega0 = {omega0} (中間軸付近、ジャニベコフ体制)\n")

    _, quats_true, omegas_true = integrate_trajectory(q0, omega0, dt, n_steps, true_inertia, "euler")
    flips = np.sum(np.diff(np.sign(omegas_true[:, 1])) != 0)
    ok = check("ジャニベコフ体制の軌道で反転が実際に起きている", flips > 0, f"flips={flips}")
    all_passed &= ok
    print()

    # --- ノイズ付き観測を1系列だけ生成し、全ての慣性設定で使い回す (公平な比較) ---
    rng_meas = np.random.default_rng(5)
    measurements = [noisy_measurement(quats_true[k], rng_meas, meas_noise_deg) for k in range(len(quats_true))]
    rng_init = np.random.default_rng(7)
    omega0_guess = omega0 + rng_init.normal(scale=0.3, size=3)

    raw_errs = np.array([
        symmetry_aware_angle_error_deg(measurements[k], quats_true[k]) for k in range(1, len(quats_true))
    ])
    raw_mean, raw_median = raw_errs.mean(), np.median(raw_errs)

    # --- (i) 全軸一様スケーリングの誤差 ---
    uniform_pcts = [0, 10, -10, 20, -20, 30, -30, 50, -50]
    print("--- (i) 全3軸を同じ倍率でスケーリング (uniform mismatch) ---")
    uniform_results = {}
    for pct in uniform_pcts:
        scale = 1.0 + pct / 100.0
        inertia_wrong = true_inertia * scale
        errs = run_ekf_with_inertia(quats_true, measurements, dt, inertia_wrong, meas_noise_deg, omega0_guess)
        uniform_results[pct] = (errs.mean(), np.median(errs), errs.max())
        print(f"  誤差{pct:+4d}%  mean={errs.mean():6.2f}°  median={np.median(errs):6.2f}°  max={errs.max():7.2f}°")
    print()

    # --- (ii) 中間軸 (2軸目, I2=10) のみを誤らせる (Dzhanibekov に最も敏感なはず) ---
    intermediate_pcts = [0, 10, -10, 20, -20, 30, -30, 50, -50]
    print("--- (ii) 中間軸 (I2) のみをスケーリング (intermediate-axis-only mismatch) ---")
    intermediate_results = {}
    for pct in intermediate_pcts:
        scale = 1.0 + pct / 100.0
        inertia_wrong = true_inertia.copy()
        inertia_wrong[1] *= scale
        errs = run_ekf_with_inertia(quats_true, measurements, dt, inertia_wrong, meas_noise_deg, omega0_guess)
        intermediate_results[pct] = (errs.mean(), np.median(errs), errs.max())
        print(f"  誤差{pct:+4d}%  mean={errs.mean():6.2f}°  median={np.median(errs):6.2f}°  max={errs.max():7.2f}°")
    print()

    exact_mean, exact_median, _ = uniform_results[0]

    print("=" * 78)
    print("  サマリー表: 慣性誤差率 -> EKF事後誤差 (raw測定/厳密慣性ベースラインと比較)")
    print("=" * 78)
    header = f"{'inertia err %':>14} | {'uniform mean':>13} {'uniform med':>12} | {'I2-only mean':>13} {'I2-only med':>12}"
    print(header)
    print("-" * len(header))
    all_pcts = sorted(set(uniform_pcts) | set(intermediate_pcts))
    for pct in all_pcts:
        u = uniform_results.get(pct)
        i = intermediate_results.get(pct)
        u_str = f"{u[0]:13.2f} {u[1]:12.2f}" if u else f"{'--':>13} {'--':>12}"
        i_str = f"{i[0]:13.2f} {i[1]:12.2f}" if i else f"{'--':>13} {'--':>12}"
        print(f"{pct:>13d}% | {u_str} | {i_str}")
    print("-" * len(header))
    print(f"{'raw meas.':>13}  | mean={raw_mean:.2f}°  median={raw_median:.2f}°  (ベースライン: フィルタなし)")
    print(f"{'exact (0%)':>13}  | mean={exact_mean:.2f}°  median={exact_median:.2f}°  (ベースライン: 厳密慣性)")
    print()

    # --- 自動チェック: 実用的な誤差レベル (±10%, ±20%) で妥当に機能するか ---
    ok = check(
        "厳密慣性(0%)のEKFがraw測定より有意に良い",
        exact_mean < raw_mean,
        f"exact={exact_mean:.2f}°, raw={raw_mean:.2f}°",
    )
    all_passed &= ok
    for pct in (10, -10, 20, -20):
        u_mean = uniform_results[pct][0]
        ok = check(
            f"uniform {pct:+d}% 誤差でもEKFがraw測定より良い",
            u_mean < raw_mean,
            f"ekf={u_mean:.2f}°, raw={raw_mean:.2f}°",
        )
        all_passed &= ok
        i_mean = intermediate_results[pct][0]
        ok = check(
            f"I2-only {pct:+d}% 誤差でもEKFがraw測定より良い",
            i_mean < raw_mean,
            f"ekf={i_mean:.2f}°, raw={raw_mean:.2f}°",
        )
        all_passed &= ok
    print()
    for pct in (50, -50):
        u_mean = uniform_results[pct][0]
        i_mean = intermediate_results[pct][0]
        check(
            f"[参考,非PASS/FAIL] uniform {pct:+d}% 誤差でのEKF vs raw",
            True,
            f"ekf={u_mean:.2f}°, raw={raw_mean:.2f}° -> {'EKFが依然優位' if u_mean < raw_mean else 'EKFがrawより悪化'}",
        )
        check(
            f"[参考,非PASS/FAIL] I2-only {pct:+d}% 誤差でのEKF vs raw",
            True,
            f"ekf={i_mean:.2f}°, raw={raw_mean:.2f}° -> {'EKFが依然優位' if i_mean < raw_mean else 'EKFがrawより悪化'}",
        )

    print()
    print("=" * 40)
    if all_passed:
        print(f"  {PASS} 全テスト通過 (±10%/±20%の慣性誤差でもEKFはraw測定より優位)")
    else:
        print(f"  {FAIL} 失敗あり (実用的な慣性誤差レベルでEKFが劣化)")
    print("=" * 40)


if __name__ == "__main__":
    main()
