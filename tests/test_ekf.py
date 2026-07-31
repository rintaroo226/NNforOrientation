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

DT = 1.0 / 60.0
MEAS_NOISE_DEG = 2.0


def noisy_measurement(q_true: np.ndarray, rng: np.random.Generator, noise_deg: float = MEAS_NOISE_DEG) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.normal(0, noise_deg))
    return quat_mul(q_true, quat_from_rotvec(axis * angle))


def noisy_measurement_random_branch(q_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sym = BOX_SYMMETRIES[rng.integers(0, 4)]
    return noisy_measurement(quat_mul(q_true, sym), rng)


class TestScenarioA_ConstantOmegaConverges:
    """ω一定モデル、ガウス性小ノイズのみ → 事後誤差が観測ノイズより小さく収束する。"""

    def setup_trajectory(self):
        inertia = principal_inertia((3.0, 2.0, 1.0))
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega0 = np.array([0.3, -0.2, 0.5])
        _, quats, _ = integrate_trajectory(q0, omega0, DT, 300, inertia, "constant")
        return quats, omega0

    def test_posterior_error_below_measurement_noise(self):
        quats, omega0 = self.setup_trajectory()
        rng = np.random.default_rng(0)
        config = EKFConfig(process_model="constant", meas_noise_deg=MEAS_NOISE_DEG, gate_mode="both")
        ekf = BoxOrientationEKF(quats[0], np.zeros(3), np.eye(6) * 0.1, config)

        raw_errs, ekf_errs = [], []
        for k in range(1, len(quats)):
            ekf.predict(DT)
            z = noisy_measurement(quats[k], rng)
            ekf.update(z)
            q_est, _, _ = ekf.state()
            raw_errs.append(symmetry_aware_angle_error_deg(z, quats[k]))
            ekf_errs.append(symmetry_aware_angle_error_deg(q_est, quats[k]))

        assert np.mean(ekf_errs[-100:]) < np.mean(raw_errs[-100:])

    def test_omega_converges_to_truth(self):
        quats, omega0 = self.setup_trajectory()
        rng = np.random.default_rng(0)
        config = EKFConfig(process_model="constant", meas_noise_deg=MEAS_NOISE_DEG, gate_mode="both")
        ekf = BoxOrientationEKF(quats[0], np.zeros(3), np.eye(6) * 0.1, config)
        for k in range(1, len(quats)):
            ekf.predict(DT)
            ekf.update(noisy_measurement(quats[k], rng))
        assert np.linalg.norm(ekf.omega - omega0) < 0.05


class TestScenarioB_OutlierGating:
    """外れ値混入時、ゲーティングON/OFFの対照実験。"""

    def run(self, gate_mode: str, quats: np.ndarray, outlier_step: int):
        rng_meas = np.random.default_rng(2)
        rng_outlier = np.random.default_rng(2)
        config = EKFConfig(process_model="constant", meas_noise_deg=MEAS_NOISE_DEG, gate_mode=gate_mode)
        ekf = BoxOrientationEKF(quats[0], np.zeros(3), np.eye(6) * 0.05, config)
        errs = []
        accepted_at_outlier = False
        for k in range(1, len(quats)):
            ekf.predict(DT)
            if k == outlier_step:
                z = sample_uniform_quat(rng_outlier)
            else:
                z = noisy_measurement(quats[k], rng_meas)
            result = ekf.update(z)
            if k == outlier_step:
                accepted_at_outlier = result.accepted
            q_est, _, _ = ekf.state()
            errs.append(symmetry_aware_angle_error_deg(q_est, quats[k]))
        return np.array(errs), accepted_at_outlier

    def setup_trajectory(self):
        inertia = principal_inertia((3.0, 2.0, 1.0))
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega0 = np.array([0.3, -0.2, 0.5])
        _, quats, _ = integrate_trajectory(q0, omega0, DT, 100, inertia, "constant")
        return quats

    def test_gated_rejects_outlier(self):
        quats = self.setup_trajectory()
        _, accepted = self.run("both", quats, outlier_step=50)
        assert accepted is False

    def test_ungated_accepts_outlier_negative_control(self):
        """ゲートを外すと同じ外れ値が採用されることを確認する (ゲートが無意味に効いていないことの検証)。"""
        quats = self.setup_trajectory()
        _, accepted = self.run("none", quats, outlier_step=50)
        assert accepted is True

    def test_gated_error_stays_small_ungated_spikes(self):
        quats = self.setup_trajectory()
        errs_gated, _ = self.run("both", quats, outlier_step=50)
        errs_ungated, _ = self.run("none", quats, outlier_step=50)
        err_gated = errs_gated[49]
        err_ungated = errs_ungated[49]
        assert err_gated < 2.0
        assert err_ungated > 3 * err_gated


class TestScenarioC_SymmetryBranchTracking:
    """観測が毎回ランダムな対称分岐で来ても、予測値基準の分岐選択で誤差スパイクが起きない。"""

    def test_no_error_spike_with_random_branch_measurements(self):
        inertia = principal_inertia((3.0, 2.0, 1.0))
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega0 = np.array([0.4, 0.3, -0.2])
        _, quats, _ = integrate_trajectory(q0, omega0, DT, 200, inertia, "constant")

        rng = np.random.default_rng(3)
        config = EKFConfig(process_model="constant", meas_noise_deg=MEAS_NOISE_DEG, gate_mode="both")
        ekf = BoxOrientationEKF(quats[0], np.zeros(3), np.eye(6) * 0.1, config)
        errs = []
        for k in range(1, len(quats)):
            ekf.predict(DT)
            z = noisy_measurement_random_branch(quats[k], rng)
            ekf.update(z)
            q_est, _, _ = ekf.state()
            errs.append(symmetry_aware_angle_error_deg(q_est, quats[k]))

        steady = np.array(errs[20:])  # warm-up を除いた定常状態
        assert steady.max() < 5.0


class TestScenarioD_EulerVsConstantOnDzhanibekovTrajectory:
    """中間軸付近のタンブリング(ジャニベコフ体制)で euler モデルが constant モデルより
    明確に優れることを確認する。Phase3 でオイラー方程式に切り替える動機そのものの
    自動回帰テスト。

    (初期角速度の推定値には現実的な誤差を持たせる。ゼロから始める完全なコールド
    スタートだと、真の角速度が学習される前に姿勢予測が大きくずれ、対称分岐の
    誤判定という別の問題を引き起こしてしまい、constant/euler の比較にならない。)
    """

    def build_trajectory(self):
        inertia = principal_inertia((3.0, 2.0, 1.0))
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega0 = np.array([0.05, 3.0, 0.05])  # 中間軸(y, I=10)付近
        n_steps = 900  # 15秒、この間に反転が複数回起きる
        _, quats, omegas = integrate_trajectory(q0, omega0, DT, n_steps, inertia, "euler")
        return inertia, quats, omegas, omega0

    def test_ground_truth_actually_flips(self):
        _, _, omegas, _ = self.build_trajectory()
        flips = np.sum(np.diff(np.sign(omegas[:, 1])) != 0)
        assert flips > 0

    def run_filter(self, process_model: str, inertia, quats, measurements, omega0_guess):
        config = EKFConfig(
            process_model=process_model,
            inertia=inertia if process_model == "euler" else None,
            meas_noise_deg=MEAS_NOISE_DEG,
            gate_mode="both",
        )
        p0 = np.eye(6)
        p0[:3, :3] *= 0.3
        p0[3:, 3:] *= 0.5
        ekf = BoxOrientationEKF(quats[0], omega0_guess, p0, config)
        errs = []
        for k in range(1, len(quats)):
            ekf.predict(DT)
            ekf.update(measurements[k])
            q_est, _, _ = ekf.state()
            errs.append(symmetry_aware_angle_error_deg(q_est, quats[k]))
        return np.array(errs)

    def test_euler_model_tracks_far_better_than_constant_model(self):
        inertia, quats, _, omega0 = self.build_trajectory()
        rng_meas = np.random.default_rng(5)
        measurements = [noisy_measurement(quats[k], rng_meas) for k in range(len(quats))]
        omega0_guess = omega0 + np.random.default_rng(7).normal(scale=0.3, size=3)

        errs_const = self.run_filter("constant", inertia, quats, measurements, omega0_guess)
        errs_euler = self.run_filter("euler", inertia, quats, measurements, omega0_guess)

        assert errs_euler.mean() < errs_const.mean() / 5
        assert errs_euler.mean() < 3.0
