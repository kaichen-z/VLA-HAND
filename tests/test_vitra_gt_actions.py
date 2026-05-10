import numpy as np
import pytest

from scripts.evaluate_vitra_gt_actions import masked_sse_and_count, per_step_sse_and_count, split_hand_metrics


def test_masked_sse_and_count_uses_scalar_action_mask():
    diff = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    mask = np.array([[True, False, True], [False, True, False]])

    sse, count = masked_sse_and_count(diff, mask)

    assert sse == pytest.approx(1.0 + 9.0 + 25.0)
    assert count == 3.0


def test_split_hand_metrics_halves_even_action_dimension():
    diff = np.array([[[1.0, 2.0, 10.0, 20.0]]], dtype=np.float32)
    mask = np.ones_like(diff, dtype=bool)

    metrics = split_hand_metrics(diff, mask)

    assert metrics["left_action_mse"] == pytest.approx((1.0 + 4.0) / 2.0)
    assert metrics["right_action_mse"] == pytest.approx((100.0 + 400.0) / 2.0)
    assert metrics["left_valid_scalar_count"] == 2
    assert metrics["right_valid_scalar_count"] == 2


def test_per_step_sse_and_count_returns_stepwise_totals():
    diff = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, False], [True, True]])

    sse, count = per_step_sse_and_count(diff, mask)

    assert sse.tolist() == pytest.approx([1.0, 9.0 + 16.0])
    assert count.tolist() == [1.0, 2.0]
