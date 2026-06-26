import math

import torch
import pytest

from silhouette_pose.losses import (
    _BOX_SYMMETRIES,
    quat_mul,
    quaternion_angle_error_deg,
    symmetry_aware_angle_error_deg,
    symmetry_aware_quaternion_loss,
)

ATOL = 1e-4


def rand_unit_quats(n: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    q = torch.randn(n, 4)
    return q / q.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# quat_mul
# ---------------------------------------------------------------------------

class TestQuatMul:
    def test_identity_left(self):
        q = rand_unit_quats(8)
        identity = torch.tensor([[1., 0., 0., 0.]]).expand(8, -1)
        result = quat_mul(identity, q)
        assert torch.allclose(result, q, atol=ATOL)

    def test_identity_right(self):
        q = rand_unit_quats(8)
        identity = torch.tensor([[1., 0., 0., 0.]]).expand(8, -1)
        result = quat_mul(q, identity)
        assert torch.allclose(result, q, atol=ATOL)

    def test_rx180_ry180_equals_rz180(self):
        """直方体対称群が閉じていること: Rx180 ⊗ Ry180 = Rz180."""
        rx = torch.tensor([[0., 1., 0., 0.]])
        ry = torch.tensor([[0., 0., 1., 0.]])
        rz = torch.tensor([[0., 0., 0., 1.]])
        result = quat_mul(rx, ry)
        # q と -q は同一回転なので abs で比較
        assert torch.allclose(result.abs(), rz.abs(), atol=ATOL)

    def test_symmetry_group_closed(self):
        """4 要素すべての積がグループ内に収まること。"""
        syms = _BOX_SYMMETRIES
        for i in range(4):
            for j in range(4):
                product = quat_mul(syms[i].unsqueeze(0), syms[j].unsqueeze(0)).squeeze(0)
                in_group = any(
                    torch.allclose(product.abs(), syms[k].abs(), atol=ATOL)
                    for k in range(4)
                )
                assert in_group, f"syms[{i}] ⊗ syms[{j}] = {product} はグループ外"


# ---------------------------------------------------------------------------
# symmetry_aware_angle_error_deg
# ---------------------------------------------------------------------------

class TestSymmetryAwareAngleError:
    def test_pred_equals_target_gives_zero(self):
        """1. pred == target のとき 0 deg。"""
        q = rand_unit_quats(16)
        angles = symmetry_aware_angle_error_deg(q, q)
        assert torch.allclose(angles, torch.zeros(16), atol=ATOL), angles

    def test_neg_target_gives_zero(self):
        """2. pred == -target のときも 0 deg (q と -q は同一回転)。"""
        q = rand_unit_quats(16)
        angles = symmetry_aware_angle_error_deg(q, -q)
        assert torch.allclose(angles, torch.zeros(16), atol=ATOL), angles

    @pytest.mark.parametrize("sym_idx, label", [
        (1, "Rx180"),
        (2, "Ry180"),
        (3, "Rz180"),
    ])
    def test_sym_equiv_gives_zero(self, sym_idx: int, label: str):
        """3-4. pred == target ⊗ R{x,y,z}(180°) のとき symmetry-aware は 0 deg。"""
        target = rand_unit_quats(16)
        sym = _BOX_SYMMETRIES[sym_idx].unsqueeze(0)          # [1, 4]
        pred = quat_mul(target, sym.expand(16, -1))           # target ⊗ sym_i

        # 通常のangle errorは 0 でないことを確認 (テストの前提)
        normal_angles = quaternion_angle_error_deg(pred, target)
        assert not torch.allclose(normal_angles, torch.zeros(16), atol=1.0), \
            f"{label}: 通常 angle error が 0 になっている (対称性が偶然一致?)"

        # symmetry-aware は 0 になること
        sym_angles = symmetry_aware_angle_error_deg(pred, target)
        assert torch.allclose(sym_angles, torch.zeros(16), atol=ATOL), \
            f"{label}: sym_angles={sym_angles}"

    def test_random_pred_has_large_error(self):
        """ランダム予測の平均誤差が対称性考慮後も有意に大きいこと (> 30 deg)。"""
        pred = rand_unit_quats(1000, seed=1)
        target = rand_unit_quats(1000, seed=2)
        mean_angle = symmetry_aware_angle_error_deg(pred, target).mean().item()
        assert mean_angle > 30.0, f"mean_angle={mean_angle:.2f} deg"


# ---------------------------------------------------------------------------
# symmetry_aware_quaternion_loss
# ---------------------------------------------------------------------------

class TestSymmetryAwareLoss:
    def test_perfect_pred_gives_small_loss(self):
        """pred == target のとき loss が -log(eps) に近い (非常に小さい負値)。"""
        q = rand_unit_quats(16)
        loss = symmetry_aware_quaternion_loss(q, q)
        assert loss.item() < -5.0, f"loss={loss.item()}"

    def test_sym_equiv_gives_same_loss_as_identity(self):
        """pred == target ⊗ sym のとき、pred == target と同じ loss。"""
        target = rand_unit_quats(32)
        loss_identity = symmetry_aware_quaternion_loss(target, target)

        for sym_idx in range(1, 4):
            sym = _BOX_SYMMETRIES[sym_idx].unsqueeze(0).expand(32, -1)
            pred_equiv = quat_mul(target, sym)
            loss_equiv = symmetry_aware_quaternion_loss(pred_equiv, target)
            assert torch.allclose(loss_identity, loss_equiv, atol=ATOL), \
                f"sym[{sym_idx}]: loss_identity={loss_identity:.6f}, loss_equiv={loss_equiv:.6f}"

    def test_loss_is_worse_for_random(self):
        """ランダム予測は完全一致より loss が大きい (0 に近い)。"""
        q = rand_unit_quats(64)
        pred_rand = rand_unit_quats(64, seed=99)
        loss_perfect = symmetry_aware_quaternion_loss(q, q)
        loss_random = symmetry_aware_quaternion_loss(pred_rand, q)
        assert loss_random > loss_perfect, \
            f"loss_perfect={loss_perfect:.4f}, loss_random={loss_random:.4f}"
