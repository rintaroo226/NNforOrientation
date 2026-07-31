import math

import numpy as np
import pytest

from quat_np import BOX_SYMMETRIES
from rigid_body import (
    euler_rhs,
    integrate_trajectory,
    principal_inertia,
    rk4_step,
    state_derivative,
)

ATOL = 1e-4


def test_principal_inertia_matches_box_3_2_1():
    inertia = principal_inertia((3.0, 2.0, 1.0))
    assert np.allclose(inertia, [5, 10, 13])


class TestConstantModel:
    def test_omega_unchanged(self):
        inertia = principal_inertia((3.0, 2.0, 1.0))
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega0 = np.array([2.0, 0.0, 0.0])
        _, _, omegas = integrate_trajectory(q0, omega0, dt=1e-3, n_steps=500,
                                             inertia=inertia, dynamics="constant")
        assert np.allclose(omegas, omega0, atol=1e-8)

    def test_half_period_x_axis_spin_hits_symmetry(self):
        """主軸(x)ぴったりの一定回転は、半周期後に Rx(180°) と一致する
        (直方体の対称性そのものが 180° フリップであることの確認)。"""
        inertia = principal_inertia((3.0, 2.0, 1.0))
        w = 2.0
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega0 = np.array([w, 0.0, 0.0])
        n_steps = 2000
        dt = (math.pi / w) / n_steps
        _, quats, _ = integrate_trajectory(q0, omega0, dt, n_steps, inertia, "constant")
        q_final = quats[-1]
        assert any(
            np.allclose(np.abs(q_final), np.abs(BOX_SYMMETRIES[i]), atol=1e-3)
            for i in range(4)
        )

    def test_quaternion_stays_unit_norm(self):
        inertia = principal_inertia((3.0, 2.0, 1.0))
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega0 = np.array([0.3, 0.7, -0.5])
        _, quats, _ = integrate_trajectory(q0, omega0, dt=1e-3, n_steps=1000,
                                            inertia=inertia, dynamics="constant")
        assert np.allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-6)


class TestEulerModel:
    def test_energy_and_angular_momentum_conserved(self):
        inertia = principal_inertia((3.0, 2.0, 1.0))
        rng = np.random.default_rng(1)
        omega0 = rng.normal(size=3) * 1.5
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        _, _, omegas = integrate_trajectory(q0, omega0, dt=1e-3, n_steps=5000,
                                             inertia=inertia, dynamics="euler")
        energy = 0.5 * np.sum(inertia * omegas ** 2, axis=1)
        l_mag2 = np.sum((inertia * omegas) ** 2, axis=1)
        assert (energy.max() - energy.min()) / energy[0] < 1e-4
        assert (l_mag2.max() - l_mag2.min()) / l_mag2[0] < 1e-4

    @pytest.mark.parametrize("axis_idx,label", [(0, "minor"), (2, "major")])
    def test_stable_axis_spin_has_no_flip(self, axis_idx: int, label: str):
        """最大軸・最小軸まわりのスピンは安定 (角速度の符号反転が起きない)。"""
        inertia = principal_inertia((3.0, 2.0, 1.0))
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega0 = np.full(3, 1e-3)
        omega0[axis_idx] = 5.0
        _, _, omegas = integrate_trajectory(q0, omega0, dt=2e-3, n_steps=20000,
                                             inertia=inertia, dynamics="euler")
        flips = np.sum(np.diff(np.sign(omegas[:, axis_idx])) != 0)
        assert flips == 0, f"{label} axis should be stable, got {flips} flips"

    def test_intermediate_axis_spin_flips(self):
        """ジャニベコフ効果: 中間軸(I=10, y軸)付近のスピンは不安定で反転する。"""
        inertia = principal_inertia((3.0, 2.0, 1.0))
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega0 = np.array([1e-3, 5.0, 1e-3])
        _, _, omegas = integrate_trajectory(q0, omega0, dt=2e-3, n_steps=20000,
                                             inertia=inertia, dynamics="euler")
        flips = np.sum(np.diff(np.sign(omegas[:, 1])) != 0)
        assert flips > 0, "expected Dzhanibekov-effect flip on the intermediate axis"


class TestJacobianRobustness:
    def test_central_difference_stable_across_epsilon(self):
        """数値微分ヤコビアンが epsilon の取り方に対してロバストであること。"""
        inertia = principal_inertia((3.0, 2.0, 1.0))
        state0 = np.array([1.0, 0.0, 0.0, 0.0, 0.5, 1.2, -0.3])

        def numeric_jacobian(eps: float) -> np.ndarray:
            n = len(state0)
            jac = np.zeros((n, n))
            for i in range(n):
                d = np.zeros(n)
                d[i] = eps
                f_plus = state_derivative(state0 + d, inertia, "euler")
                f_minus = state_derivative(state0 - d, inertia, "euler")
                jac[:, i] = (f_plus - f_minus) / (2 * eps)
            return jac

        jac_a = numeric_jacobian(1e-4)
        jac_b = numeric_jacobian(1e-6)
        assert np.max(np.abs(jac_a - jac_b)) < 1e-3


def test_euler_rhs_zero_for_principal_axis_spin():
    """主軸ぴったりのスピンでは dω/dt = 0 (角速度の向きが変化しない)。"""
    inertia = principal_inertia((3.0, 2.0, 1.0))
    for omega in [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]:
        assert np.allclose(euler_rhs(omega, inertia), 0.0, atol=ATOL)


def test_rk4_step_renormalizes_quaternion():
    inertia = principal_inertia((3.0, 2.0, 1.0))
    state = np.array([1.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.1])
    new_state = rk4_step(state, dt=0.01, inertia=inertia, dynamics="euler")
    assert abs(np.linalg.norm(new_state[:4]) - 1.0) < 1e-8
