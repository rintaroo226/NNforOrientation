"""ekf.py (直方体姿勢+角速度の EKF) の動作確認スクリプト (torch/MATLAB/チェックポイント不要)。

実行: python3 verify_ekf.py
"""
import numpy as np

from ekf import BoxOrientationEKF, EKFConfig
from quat_np import (
    BOX_SYMMETRIES,
    quat_from_rotvec,
    quat_mul,
    sample_uniform_quat,
    symmetry_aware_angle_error_deg,
)
from rigid_body import integrate_trajectory, principal_inertia

ATOL = 1e-4
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


def noisy_measurement_random_branch(
    q_true: np.ndarray, rng: np.random.Generator, noise_deg: float
) -> np.ndarray:
    """直方体対称性のため、NN 測定はどの等価分岐で出てもおかしくない状況を模擬する。"""
    sym = BOX_SYMMETRIES[rng.integers(0, 4)]
    q_branch = quat_mul(q_true, sym)
    return noisy_measurement(q_branch, rng, noise_deg)


def main() -> None:
    all_passed = True
    inertia = principal_inertia((3.0, 2.0, 1.0))
    dt = 1.0 / 60.0
    meas_noise_deg = 2.0

    print("=== 動作確認: ekf.py (BoxOrientationEKF) ===\n")

    # --- シナリオA: ω一定+ガウス性小ノイズ → 事後誤差が観測ノイズより有意に収束 ---
    print("シナリオA: ω一定モデル、観測ノイズのみ → 事後誤差が観測誤差より小さく収束")
    rng = np.random.default_rng(0)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    omega0 = np.array([0.3, -0.2, 0.5])
    n_steps = 300
    _, quats, _ = integrate_trajectory(q0, omega0, dt, n_steps, inertia, "constant")

    config = EKFConfig(process_model="constant", meas_noise_deg=meas_noise_deg, gate_mode="both")
    ekf = BoxOrientationEKF(quats[0], np.zeros(3), np.eye(6) * 0.1, config)
    raw_errs, ekf_errs = [], []
    for k in range(1, n_steps + 1):
        ekf.predict(dt)
        z = noisy_measurement(quats[k], rng, meas_noise_deg)
        ekf.update(z)
        q_est, omega_est, _ = ekf.state()
        raw_errs.append(symmetry_aware_angle_error_deg(z, quats[k]))
        ekf_errs.append(symmetry_aware_angle_error_deg(q_est, quats[k]))
    raw_tail = np.mean(raw_errs[-100:])
    ekf_tail = np.mean(ekf_errs[-100:])
    ok = check(
        "事後誤差(定常状態)が観測誤差より小さい",
        ekf_tail < raw_tail,
        f"raw={raw_tail:.2f}°, ekf={ekf_tail:.2f}°",
    )
    all_passed &= ok
    omega_err = np.linalg.norm(ekf.omega - omega0)
    ok = check("角速度ωが真値近くに収束する", omega_err < 0.05, f"|Δω|={omega_err:.4f}")
    all_passed &= ok
    print()

    # --- シナリオB: 大外れ値を1つ混入 → ゲーティングON/OFFの対照実験 ---
    print("シナリオB: 外れ値混入 → ゲーティングONで抑制、OFFで悪化 (対照実験)")
    outlier_step = 50
    n_steps_b = 100
    _, quats_b, _ = integrate_trajectory(q0, omega0, dt, n_steps_b, inertia, "constant")

    def run_scenario_b(gate_mode: str) -> tuple[np.ndarray, bool]:
        rng_local = np.random.default_rng(2)
        rng_outlier = np.random.default_rng(2)
        cfg = EKFConfig(process_model="constant", meas_noise_deg=meas_noise_deg, gate_mode=gate_mode)
        f = BoxOrientationEKF(quats_b[0], np.zeros(3), np.eye(6) * 0.05, cfg)
        errs = []
        accepted_at_outlier = False
        for k in range(1, n_steps_b + 1):
            f.predict(dt)
            if k == outlier_step:
                z = sample_uniform_quat(rng_outlier)
            else:
                z = noisy_measurement(quats_b[k], rng_local, meas_noise_deg)
            result = f.update(z)
            if k == outlier_step:
                accepted_at_outlier = result.accepted
            q_est, _, _ = f.state()
            errs.append(symmetry_aware_angle_error_deg(q_est, quats_b[k]))
        return np.array(errs), accepted_at_outlier

    errs_gated, accepted_gated = run_scenario_b("both")
    errs_ungated, accepted_ungated = run_scenario_b("none")
    ok = check("ゲーティングONで外れ値が棄却される", not accepted_gated)
    all_passed &= ok
    ok = check("ゲーティングOFFで外れ値が採用される (対照実験)", accepted_ungated)
    all_passed &= ok
    err_gated_at_outlier = errs_gated[outlier_step - 1]
    err_ungated_at_outlier = errs_ungated[outlier_step - 1]
    ok = check(
        "ゲーティングONなら事後誤差が小さく保たれる",
        err_gated_at_outlier < 2.0,
        f"err={err_gated_at_outlier:.2f}°",
    )
    all_passed &= ok
    ok = check(
        "ゲーティングOFFなら事後誤差が明確に悪化する (無意味なゲートでないことの確認)",
        err_ungated_at_outlier > 3 * err_gated_at_outlier,
        f"gated={err_gated_at_outlier:.2f}°, ungated={err_ungated_at_outlier:.2f}°",
    )
    all_passed &= ok
    print()

    # --- シナリオC: 対称分岐が毎回ランダムに変わる観測列 ---
    print("シナリオC: 観測が毎回ランダムな対称分岐で来ても誤差スパイクなく追従できる")
    rng = np.random.default_rng(3)
    omega0_c = np.array([0.4, 0.3, -0.2])
    n_steps_c = 200
    _, quats_c, _ = integrate_trajectory(q0, omega0_c, dt, n_steps_c, inertia, "constant")
    cfg = EKFConfig(process_model="constant", meas_noise_deg=meas_noise_deg, gate_mode="both")
    ekf_c = BoxOrientationEKF(quats_c[0], np.zeros(3), np.eye(6) * 0.1, cfg)
    errs_c = []
    for k in range(1, n_steps_c + 1):
        ekf_c.predict(dt)
        z = noisy_measurement_random_branch(quats_c[k], rng, meas_noise_deg)
        ekf_c.update(z)
        q_est, _, _ = ekf_c.state()
        errs_c.append(symmetry_aware_angle_error_deg(q_est, quats_c[k]))
    errs_c = np.array(errs_c)
    steady = errs_c[20:]  # 収束後 (warm-up 除く)
    ok = check(
        "ランダム分岐でも定常状態で誤差スパイクが起きない (max < 5°)",
        steady.max() < 5.0,
        f"max={steady.max():.2f}°, mean={steady.mean():.2f}°",
    )
    all_passed &= ok
    print()

    # --- シナリオD: ジャニベコフ体制で constant モデルと euler モデルを比較 ---
    # (これが Phase3 でオイラー方程式に切り替える動機そのものの自動回帰テスト)
    print("シナリオD: 中間軸付近のタンブリング(ジャニベコフ体制)で euler モデルが constant モデルより明確に優れる")
    from rigid_body import integrate_trajectory as _integrate

    omega0_d = np.array([0.05, 3.0, 0.05])  # 中間軸(y, I=10)付近
    n_steps_d = 900  # 15秒、この間に反転が2回起きる
    _, quats_d, omegas_d = _integrate(q0, omega0_d, dt, n_steps_d, inertia, "euler")
    flips = np.sum(np.diff(np.sign(omegas_d[:, 1])) != 0)
    ok = check("ジャニベコフ体制の軌道で反転が実際に起きている", flips > 0, f"flips={flips}")
    all_passed &= ok

    rng_meas = np.random.default_rng(5)
    measurements_d = [
        noisy_measurement(quats_d[k], rng_meas, meas_noise_deg) for k in range(len(quats_d))
    ]
    rng_init = np.random.default_rng(7)
    omega0_guess = omega0_d + rng_init.normal(scale=0.3, size=3)

    def run_scenario_d(process_model: str) -> np.ndarray:
        cfg = EKFConfig(
            process_model=process_model,
            inertia=inertia if process_model == "euler" else None,
            meas_noise_deg=meas_noise_deg,
            gate_mode="both",
        )
        p0 = np.eye(6)
        p0[:3, :3] *= 0.3
        p0[3:, 3:] *= 0.5
        f = BoxOrientationEKF(quats_d[0], omega0_guess, p0, cfg)
        errs = []
        for k in range(1, len(quats_d)):
            f.predict(dt)
            f.update(measurements_d[k])
            q_est, _, _ = f.state()
            errs.append(symmetry_aware_angle_error_deg(q_est, quats_d[k]))
        return np.array(errs)

    errs_const = run_scenario_d("constant")
    errs_euler = run_scenario_d("euler")
    ok = check(
        "euler モデルの平均誤差が constant モデルより明確に小さい",
        errs_euler.mean() < errs_const.mean() / 5,
        f"constant={errs_const.mean():.2f}°, euler={errs_euler.mean():.2f}°",
    )
    all_passed &= ok
    ok = check(
        "euler モデルは反転を通しても小さい誤差を保つ (mean < 3°)",
        errs_euler.mean() < 3.0,
        f"euler mean={errs_euler.mean():.2f}°, max={errs_euler.max():.2f}°",
    )
    all_passed &= ok
    print()

    print("=" * 40)
    if all_passed:
        print(f"  {PASS} 全テスト通過")
    else:
        print(f"  {FAIL} 失敗あり")
    print("=" * 40)


if __name__ == "__main__":
    main()
