import math

import numpy as np
import torch
import pytest

from silhouette_pose.losses import (
    _BOX_SYMMETRIES,
    quat_mul as quat_mul_torch,
    symmetry_aware_angle_error_deg as symmetry_aware_angle_error_deg_torch,
)
from quat_np import (
    BOX_SYMMETRIES,
    nearest_sym_quat,
    quat_conjugate,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    quat_to_matrix,
    rotvec_from_quat,
    sample_uniform_quat,
    symmetry_aware_angle_error_deg,
)

ATOL = 1e-4


def rand_unit_quats_np(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4)).astype(np.float32)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# torch 版 (silhouette_pose.losses) との数値一致
# ---------------------------------------------------------------------------

class TestMatchesTorch:
    def test_quat_mul_matches_torch(self):
        q = rand_unit_quats_np(16, seed=0)
        r = rand_unit_quats_np(16, seed=1)
        np_result = quat_mul(q, r)
        torch_result = quat_mul_torch(torch.from_numpy(q), torch.from_numpy(r)).numpy()
        assert np.allclose(np_result, torch_result, atol=ATOL)

    def test_box_symmetries_match_torch(self):
        assert np.allclose(BOX_SYMMETRIES, _BOX_SYMMETRIES.numpy(), atol=ATOL)

    def test_symmetry_aware_angle_error_matches_torch(self):
        pred = rand_unit_quats_np(32, seed=2)
        target = rand_unit_quats_np(32, seed=3)
        np_errs = np.array([
            symmetry_aware_angle_error_deg(pred[i], target[i]) for i in range(32)
        ])
        torch_errs = symmetry_aware_angle_error_deg_torch(
            torch.from_numpy(pred), torch.from_numpy(target)
        ).numpy()
        assert np.allclose(np_errs, torch_errs, atol=1e-2), (np_errs - torch_errs)


# ---------------------------------------------------------------------------
# quat_mul / quat_conjugate / quat_normalize
# ---------------------------------------------------------------------------

class TestQuatMul:
    def test_identity_left(self):
        q = rand_unit_quats_np(8)
        identity = np.tile(np.array([1., 0., 0., 0.], dtype=np.float32), (8, 1))
        assert np.allclose(quat_mul(identity, q), q, atol=ATOL)

    def test_conjugate_inverts(self):
        q = rand_unit_quats_np(8)
        prod = quat_mul(q, quat_conjugate(q))
        identity = np.tile(np.array([1., 0., 0., 0.], dtype=np.float32), (8, 1))
        assert np.allclose(np.abs(prod), identity, atol=ATOL)

    def test_normalize(self):
        q = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert np.allclose(quat_normalize(q), [1.0, 0.0, 0.0, 0.0], atol=ATOL)


# ---------------------------------------------------------------------------
# nearest_sym_quat / symmetry_aware_angle_error_deg
# ---------------------------------------------------------------------------

class TestSymmetry:
    def test_pred_equals_target_gives_zero(self):
        q = rand_unit_quats_np(1)[0]
        assert symmetry_aware_angle_error_deg(q, q) < ATOL

    @pytest.mark.parametrize("sym_idx", [1, 2, 3])
    def test_sym_equiv_gives_zero_and_correct_index(self, sym_idx: int):
        target = rand_unit_quats_np(1)[0]
        pred = quat_mul(target, BOX_SYMMETRIES[sym_idx])
        equiv, best_idx = nearest_sym_quat(pred, target)
        assert best_idx == sym_idx
        assert symmetry_aware_angle_error_deg(pred, target) < ATOL

    def test_random_pred_has_large_error(self):
        preds = rand_unit_quats_np(200, seed=10)
        targets = rand_unit_quats_np(200, seed=11)
        errs = [symmetry_aware_angle_error_deg(p, t) for p, t in zip(preds, targets)]
        assert sum(errs) / len(errs) > 30.0


# ---------------------------------------------------------------------------
# rotvec_from_quat / quat_from_rotvec (log/exp map)
# ---------------------------------------------------------------------------

class TestRotvec:
    def test_identity_gives_zero_rotvec(self):
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert np.allclose(rotvec_from_quat(q), [0, 0, 0], atol=ATOL)

    def test_roundtrip(self):
        for q in rand_unit_quats_np(16, seed=4):
            if q[0] < 0:
                q = -q  # rotvec_from_quat の角度域は [0, pi] を仮定
            v = rotvec_from_quat(q)
            q2 = quat_from_rotvec(v)
            assert np.allclose(q, q2, atol=1e-3) or np.allclose(q, -q2, atol=1e-3)

    def test_known_90deg_x(self):
        v = np.array([math.pi / 2, 0.0, 0.0])
        q = quat_from_rotvec(v)
        expected = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
        assert np.allclose(q, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# quat_to_matrix
# ---------------------------------------------------------------------------

class TestQuatToMatrix:
    def test_identity_is_eye(self):
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert np.allclose(quat_to_matrix(q), np.eye(3), atol=ATOL)

    def test_is_orthonormal(self):
        for q in rand_unit_quats_np(8, seed=5):
            R = quat_to_matrix(q)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-4)
            assert abs(np.linalg.det(R) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# sample_uniform_quat
# ---------------------------------------------------------------------------

class TestSampleUniformQuat:
    def test_is_unit_norm(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = sample_uniform_quat(rng)
            assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-5
