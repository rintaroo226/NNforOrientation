"""クォータニオン推定値と(クロップ前の)bboxピクセルサイズから、レンダリング
距離を逆算する。

bboxクロップは「対象がフレームに占める割合」を常に一定に正規化することで
距離不変性を得ているが、正規化の際に捨てた倍率(=元画像でのbboxサイズ)
そのものが距離の手がかりになる。ただし同じ距離でも姿勢によって見かけの
大きさは変わる(角から見る方が面から見るより大きく見える)ため、bboxサイズ
だけでは距離は決まらない。姿勢が分かっていれば、「その姿勢の箱を基準距離に
置いたときの理論的なbboxサイズ」を計算でき、実測との比から距離を逆算できる。

事前検証(matlab/database_distance_sweep, GTクォータニオン使用)では、
箱がカメラのFOV内に完全に収まっている限り、この単純な比例(弱透視近似)で
誤差1.3%以下(距離10〜55m)。箱がフレーム端で切れている(clipping)場合は
実際より小さく見えて距離を過大評価するため、tight_bbox_touches_border で
検出・除外することを推奨する。

投影の規約(カメラ配置、world+Xを画面左に反転する符号)は
render_ekf_overlay.py の project_box_edges / matlab/renderBoxImage.m と
完全に一致させている(ただしbboxの「サイズ」は各頂点の符号反転では変わらない
ため、この符号反転はサイズ計算には実質影響しない)。
"""
import numpy as np

from quat_np import quat_to_matrix


def box_vertices(box_size: tuple[float, float, float]) -> np.ndarray:
    w, h, d = box_size
    return np.array([
        [-w/2, -h/2, -d/2], [w/2, -h/2, -d/2], [w/2, h/2, -d/2], [-w/2, h/2, -d/2],
        [-w/2, -h/2, d/2], [w/2, -h/2, d/2], [w/2, h/2, d/2], [-w/2, h/2, d/2],
    ])


def bbox_px_analytic(
    q: np.ndarray,
    distance: float,
    box_size: tuple[float, float, float],
    img_w: int = 512,
    img_h: int = 512,
    fov_deg: float = 25.0,
) -> float:
    """姿勢qの箱を距離distanceに置いたときの、投影された外接矩形の
    ピクセルサイズ(幅・高さのうち大きい方)を理論計算する(クリッピング無視)。
    """
    tan_half = np.tan(np.radians(fov_deg) / 2.0)
    v = box_vertices(box_size) @ quat_to_matrix(q).T
    v = v.copy()
    v[:, 2] += distance
    ndc_x = -v[:, 0] / (v[:, 2] * tan_half)
    ndc_y = v[:, 1] / (v[:, 2] * tan_half)
    px = (ndc_x + 1.0) / 2.0 * img_w
    py = (1.0 - (ndc_y + 1.0) / 2.0) * img_h
    return float(max(px.max() - px.min(), py.max() - py.min()))


def estimate_distance(
    q: np.ndarray,
    observed_bbox_px: float,
    box_size: tuple[float, float, float],
    ref_distance: float = 10.0,
    img_w: int = 512,
    img_h: int = 512,
    fov_deg: float = 25.0,
) -> float:
    """予測クォータニオンqと、元画像(クロップ前)でのbboxピクセルサイズから
    距離を逆算する(弱透視近似: bboxサイズ ∝ 1/distance)。

    箱がフレーム内に完全に収まっている場合のみ信頼できる。フレーム端で
    切れている(observed_bbox_pxが実際より小さい)場合は距離を過大評価する。
    """
    ref_bbox_px = bbox_px_analytic(q, ref_distance, box_size, img_w, img_h, fov_deg)
    if observed_bbox_px <= 0:
        return float("nan")
    return ref_distance * ref_bbox_px / observed_bbox_px
