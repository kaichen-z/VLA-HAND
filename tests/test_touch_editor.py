from pathlib import Path

import numpy as np
import pytest
import torch

from vitra.touch_editor.dataset import TouchEditorCacheDataset
from vitra.touch_editor.losses import touch_editor_loss
from vitra.touch_editor.model import ResidualTouchEditor
from vitra.touch_editor.cache_utils import build_future_mask, chunk_phase
from vitra.touch_editor.guidance import apply_touch_editor_once, apply_touch_guidance_schedule, seconds_to_chunk_index
from vitra.touch_editor.alignment import align_touch_to_timestamps, nearest_timestamp_indices
from vitra.touch_editor.pseudo_pair import (
    MatchFeature,
    MatchResult,
    extract_contact_verbs,
    find_best_match,
    intersect_action_masks,
)
from scripts.cache_touch_editor_base_actions import build_cache_record, resolve_checkpoint_and_config
from scripts.build_opentouch_gigahands_pseudo_pairs import replace_target_with_gigahands_match
from scripts.evaluate_touch_guided_actions import (
    add_delta_stats,
    add_hand_mse,
    filter_batch_by_paths,
    finalize_metrics,
    high_contact_path_set,
    randomize_targets_from_batch,
    shuffle_touch,
    touch_contact_score,
)
from scripts.evaluate_touch_guided_actions import parse_args as parse_eval_args
from scripts.expand_touch_editor_samples import assert_design1_record, expand_record
from scripts.summarize_touch_ablation_metrics import gain
from scripts.train_touch_editor import apply_touch_training_ablation


def write_cache_sample(path: Path, chunk_len: int = 16, edit_start_idx: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    a_base = rng.normal(size=(chunk_len, 192)).astype(np.float32)
    a_target = a_base.copy()
    a_target[edit_start_idx:, 51:120] += 0.1
    action_mask = np.ones((chunk_len, 192), dtype=bool)
    future_mask = build_future_mask(action_mask, edit_start_idx)
    np.savez(
        path,
        a_base=a_base,
        a_target=a_target.astype(np.float32),
        action_mask=action_mask,
        current_state=np.zeros((212,), dtype=np.float32),
        current_state_mask=np.ones((212,), dtype=bool),
        touch_pressure=np.ones((chunk_len, 2, 16, 16), dtype=np.float32),
        touch_mask=np.ones((chunk_len, 2), dtype=bool),
        future_mask=future_mask,
        edit_start_idx=np.asarray(edit_start_idx, dtype=np.int64),
        chunk_phase=chunk_phase(chunk_len),
        target_source=np.asarray("opentouch_derived"),
        touch_source=np.asarray("opentouch"),
    )


def test_touch_editor_cache_dataset_and_model_forward(tmp_path):
    write_cache_sample(tmp_path / "sample_000001.npz")
    ds = TouchEditorCacheDataset(tmp_path)
    item = ds[0]
    assert item["a_base"].shape == (16, 192)
    assert item["touch_pressure"].shape == (16, 2, 16, 16)
    assert torch.allclose(item["residual_target"], item["a_target"] - item["a_base"])

    batch = {
        "a_base": item["a_base"].unsqueeze(0),
        "current_state": item["current_state"].unsqueeze(0),
        "current_state_mask": item["current_state_mask"].unsqueeze(0),
        "touch_pressure": item["touch_pressure"].unsqueeze(0),
        "touch_mask": item["touch_mask"].unsqueeze(0),
        "chunk_phase": item["chunk_phase"].unsqueeze(0),
        "future_mask": item["future_mask"].unsqueeze(0),
        "action_mask": item["action_mask"].unsqueeze(0),
    }
    model = ResidualTouchEditor(hidden_dim=64, touch_feature_dim=32, num_layers=1)
    delta = model(**batch)
    assert delta.shape == (1, 16, 192)


def test_touch_editor_loss_returns_components(tmp_path):
    write_cache_sample(tmp_path / "sample_000001.npz")
    item = TouchEditorCacheDataset(tmp_path)[0]
    delta = torch.zeros((1, 16, 192), dtype=torch.float32)
    losses = touch_editor_loss(
        item["a_base"].unsqueeze(0),
        item["a_target"].unsqueeze(0),
        delta,
        item["action_mask"].unsqueeze(0),
        item["future_mask"].unsqueeze(0),
    )
    assert losses["loss"].ndim == 0
    assert losses["a_edit"].shape == (1, 16, 192)
    assert losses["loss_demo"] > 0
    assert torch.allclose(losses["loss_demo"], losses["loss_residual"])


def test_future_mask_keeps_executed_prefix_equal_to_base(tmp_path):
    write_cache_sample(tmp_path / "sample_000001.npz", edit_start_idx=10)
    item = TouchEditorCacheDataset(tmp_path)[0]
    delta = torch.ones((1, 16, 192), dtype=torch.float32)
    losses = touch_editor_loss(
        item["a_base"].unsqueeze(0),
        item["a_target"].unsqueeze(0),
        delta,
        item["action_mask"].unsqueeze(0),
        item["future_mask"].unsqueeze(0),
    )
    a_edit = losses["a_edit"][0]
    assert torch.allclose(a_edit[:10], item["a_base"][:10])
    assert torch.allclose(a_edit[10:], item["a_base"][10:] + 1.0)


def test_build_cache_record_schema():
    chunk_len = 16
    a_base = np.zeros((chunk_len, 192), dtype=np.float32)
    sample = {
        "action_list": np.ones((chunk_len, 192), dtype=np.float32),
        "action_mask": np.ones((chunk_len, 192), dtype=bool),
        "current_state": np.zeros((212,), dtype=np.float32),
        "current_state_mask": np.ones((212,), dtype=bool),
    }
    episode = {
        "opentouch": {
            "touch_pressure": np.arange(20, dtype=np.float32)[:, None, None, None] * np.ones((20, 2, 16, 16), dtype=np.float32),
            "touch_mask": np.ones((20, 2), dtype=bool),
            "video_timestamps": np.arange(20, dtype=np.float64) / 8.0,
            "touch_aligned_indices": np.arange(20, dtype=np.int64),
            "touch_aligned_timestamps": np.arange(20, dtype=np.float64) / 8.0,
            "touch_alignment_valid": np.ones((20,), dtype=bool),
        }
    }
    record = build_cache_record(a_base=a_base, sample=sample, episode=episode, frame_id=4, edit_start_idx=10)
    assert np.allclose(record["a_target"], sample["action_list"])
    assert record["a_target"].shape == (16, 192)
    assert np.allclose(record["residual_target"], record["a_target"] - record["a_base"])
    assert record["touch_pressure"].shape == (16, 2, 16, 16)
    assert record["action_frame_indices"].tolist() == list(range(4, 20))
    assert np.allclose(record["touch_pressure"][:, 0, 0, 0], np.arange(4, 20, dtype=np.float32))
    assert np.allclose(record["action_timestamps"], np.arange(4, 20, dtype=np.float64) / 8.0)
    assert record["touch_alignment_valid"].all()
    assert record["future_mask"][:10].sum() == 0
    assert record["future_mask"][10:].all()
    assert str(record["target_source"]) == "opentouch_derived"
    assert str(record["target_dataset"]) == "opentouch"
    assert str(record["touch_source"]) == "opentouch"
    assert "matched_gigahands_data_id" not in record
    assert "matched_episode_id" not in record


def test_build_cache_record_requires_touch_by_default():
    chunk_len = 16
    sample = {
        "action_list": np.ones((chunk_len, 192), dtype=np.float32),
        "action_mask": np.ones((chunk_len, 192), dtype=bool),
        "current_state": np.zeros((212,), dtype=np.float32),
        "current_state_mask": np.ones((212,), dtype=bool),
    }

    with pytest.raises(KeyError, match="touch_mode zeros"):
        build_cache_record(
            a_base=np.zeros((chunk_len, 192), dtype=np.float32),
            sample=sample,
            episode={"video_decode_frame": np.arange(20, dtype=np.int64)},
            frame_id=2,
            edit_start_idx=8,
        )


def test_build_cache_record_zero_touch_for_gigahands_style_episode():
    chunk_len = 16
    a_base = np.zeros((chunk_len, 192), dtype=np.float32)
    sample = {
        "action_list": np.ones((chunk_len, 192), dtype=np.float32),
        "action_mask": np.ones((chunk_len, 192), dtype=bool),
        "current_state": np.zeros((212,), dtype=np.float32),
        "current_state_mask": np.ones((212,), dtype=bool),
    }

    record = build_cache_record(
        a_base=a_base,
        sample=sample,
        episode={"video_decode_frame": np.arange(20, dtype=np.int64)},
        frame_id=2,
        edit_start_idx=8,
        touch_mode="zeros",
    )

    assert record["touch_pressure"].shape == (chunk_len, 2, 16, 16)
    assert record["touch_mask"].shape == (chunk_len, 2)
    assert np.allclose(record["touch_pressure"], 0.0)
    assert not record["touch_mask"].any()
    assert record["touch_aligned_indices"].tolist() == [-1] * chunk_len
    assert not record["touch_alignment_valid"].any()
    assert np.allclose(record["residual_target"], 1.0)
    assert str(record["target_source"]) == "opentouch_derived"


def test_expand_design1_cache_rebuilds_future_masks_without_changing_targets(tmp_path):
    source = tmp_path / "sample_000001.npz"
    write_cache_sample(source, edit_start_idx=8)
    record = dict(np.load(source, allow_pickle=False))

    assert_design1_record(record, source)
    expanded = expand_record(record, edit_start_idx=3, source_path=source)

    assert np.allclose(expanded["a_base"], record["a_base"])
    assert np.allclose(expanded["a_target"], record["a_target"])
    assert np.allclose(expanded["touch_pressure"], record["touch_pressure"])
    assert np.array_equal(expanded["touch_mask"], record["touch_mask"])
    assert np.allclose(expanded["residual_target"], expanded["a_target"] - expanded["a_base"])
    assert int(expanded["edit_start_idx"]) == 3
    assert expanded["future_mask"][:3].sum() == 0
    assert expanded["future_mask"][3:].all()
    assert str(expanded["target_source"]) == "opentouch_derived"
    assert str(expanded["touch_source"]) == "opentouch"


def test_pseudo_pair_matcher_prefers_contact_verb_match():
    source = MatchFeature(
        data_id=1,
        episode_id="opentouch",
        frame_id=5,
        instruction="Right hand: press the object.",
        phase=0.5,
        state=np.zeros((4,), dtype=np.float32),
        state_mask=np.ones((4,), dtype=bool),
        action_target=np.zeros((2, 3), dtype=np.float32),
        action_mask=np.ones((2, 3), dtype=bool),
        contact_verbs=extract_contact_verbs("press the object"),
    )
    verb_match = MatchFeature(
        data_id=2,
        episode_id="gigahands_press",
        frame_id=5,
        instruction="Press the button.",
        phase=0.5,
        state=np.ones((4,), dtype=np.float32) * 0.2,
        state_mask=np.ones((4,), dtype=bool),
        action_target=np.ones((2, 3), dtype=np.float32),
        action_mask=np.ones((2, 3), dtype=bool),
        contact_verbs=extract_contact_verbs("press the button"),
    )
    wrong_verb = MatchFeature(
        data_id=3,
        episode_id="gigahands_lift",
        frame_id=5,
        instruction="Lift the object.",
        phase=0.5,
        state=np.zeros((4,), dtype=np.float32),
        state_mask=np.ones((4,), dtype=bool),
        action_target=np.ones((2, 3), dtype=np.float32) * 2,
        action_mask=np.ones((2, 3), dtype=bool),
        contact_verbs=extract_contact_verbs("lift the object"),
    )

    result = find_best_match(source, [wrong_verb, verb_match], w_task=3.0, w_phase=1.0, w_state=1.0)

    assert result.index == 1
    assert result.task_cost == 0.0


def test_pseudo_pair_target_replacement_uses_gigahands_gt_and_intersection_mask():
    record = {
        "a_base": np.zeros((2, 4), dtype=np.float32),
        "a_target": np.ones((2, 4), dtype=np.float32) * 9,
        "residual_target": np.ones((2, 4), dtype=np.float32) * 9,
        "action_mask": np.array([[True, True, False, False], [True, True, True, False]]),
        "future_mask": np.ones((2, 4), dtype=np.float32),
        "edit_start_idx": np.asarray(1, dtype=np.int64),
    }
    matched = MatchFeature(
        data_id=42,
        episode_id="gigahands",
        frame_id=7,
        instruction="Right hand: hold the cup.",
        phase=0.25,
        state=np.zeros((4,), dtype=np.float32),
        state_mask=np.ones((4,), dtype=bool),
        action_target=np.ones((2, 4), dtype=np.float32) * 3,
        action_mask=np.array([[True, False, True, False], [True, True, False, False]]),
        contact_verbs=("hold",),
    )
    source = MatchFeature(
        data_id=5,
        episode_id="opentouch_episode",
        frame_id=3,
        instruction="Right hand: hold the block.",
        phase=0.2,
        state=np.zeros((4,), dtype=np.float32),
        state_mask=np.ones((4,), dtype=bool),
        action_target=np.zeros((2, 4), dtype=np.float32),
        action_mask=np.ones((2, 4), dtype=bool),
        contact_verbs=("hold",),
    )
    match = MatchResult(index=0, score=0.25, task_cost=0.0, phase_cost=0.1, state_cost=0.15)

    replaced = replace_target_with_gigahands_match(record, matched=matched, source=source, match=match)

    assert np.allclose(replaced["a_target"], 3.0)
    assert np.allclose(replaced["residual_target"], replaced["a_target"] - replaced["a_base"])
    assert replaced["action_mask"].tolist() == [[True, False, False, False], [True, True, False, False]]
    assert replaced["future_mask"][0].sum() == 0
    assert replaced["future_mask"][1].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert str(replaced["target_source"]) == "gigahands_matched"
    assert str(replaced["touch_source"]) == "opentouch"
    assert str(replaced["source_episode_id"]) == "opentouch_episode"
    assert str(replaced["matched_episode_id"]) == "gigahands"
    assert float(replaced["source_phase"]) == pytest.approx(0.2)
    assert float(replaced["matched_phase"]) == pytest.approx(0.25)
    assert int(replaced["matched_gigahands_data_id"]) == 42
    assert float(replaced["match_score"]) == pytest.approx(0.25)


def test_intersect_action_masks_requires_same_shape():
    with pytest.raises(ValueError, match="action mask shape mismatch"):
        intersect_action_masks(np.ones((2, 3), dtype=bool), np.ones((2, 4), dtype=bool))


def test_residual_loss_matches_action_demo_loss_under_editable_mask():
    a_base = torch.zeros((1, 4, 3), dtype=torch.float32)
    a_target = torch.ones((1, 4, 3), dtype=torch.float32)
    delta = torch.full((1, 4, 3), 0.25, dtype=torch.float32)
    action_mask = torch.ones((1, 4, 3), dtype=torch.bool)
    future_mask = torch.zeros((1, 4, 3), dtype=torch.float32)
    future_mask[:, 2:] = 1.0

    losses = touch_editor_loss(a_base, a_target, delta, action_mask, future_mask)

    assert torch.allclose(losses["loss_residual"], losses["loss_demo"])


def test_nearest_timestamp_indices_exact_nearest_and_tolerance():
    target = np.array([0.0, 0.10, 0.20], dtype=np.float64)
    source = np.array([0.0, 0.11, 0.50], dtype=np.float64)
    indices, valid = nearest_timestamp_indices(target, source, tolerance=0.02)
    assert indices.tolist() == [0, 1, -1]
    assert valid.tolist() == [True, True, False]


def test_align_touch_to_timestamps_zeroes_invalid_frames():
    touch_pressure = np.arange(3, dtype=np.float32)[:, None, None, None] * np.ones((3, 2, 16, 16), dtype=np.float32)
    touch_mask = np.ones((3, 2), dtype=bool)
    aligned_pressure, aligned_mask, indices, valid = align_touch_to_timestamps(
        touch_pressure,
        touch_mask,
        target_timestamps=np.array([0.0, 0.1, 0.2], dtype=np.float64),
        touch_timestamps=np.array([0.0, 0.11, 0.5], dtype=np.float64),
        tolerance=0.02,
    )
    assert indices.tolist() == [0, 1, -1]
    assert valid.tolist() == [True, True, False]
    assert np.allclose(aligned_pressure[:2, 0, 0, 0], [0.0, 1.0])
    assert aligned_mask[:2].all()
    assert not aligned_mask[2].any()
    assert np.allclose(aligned_pressure[2], 0.0)


def test_resolve_hf_checkpoint_and_config(tmp_path):
    files = [
        "README.md",
        "config/human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json",
        "epoch=0-step=140000.ckpt/weights.pt",
    ]

    def fake_list_repo_files(repo_id):
        assert repo_id == "LeoJiangOR/vitra-gigahands-keypoints-step140000"
        return files

    def fake_download(repo_id, filename):
        path = tmp_path / filename.replace("/", "__")
        path.write_text("{}", encoding="utf-8")
        return str(path)

    weights_path, config_path, repo_id = resolve_checkpoint_and_config(
        "LeoJiangOR/vitra-gigahands-keypoints-step140000",
        list_repo_files_fn=fake_list_repo_files,
        hf_hub_download_fn=fake_download,
    )

    assert repo_id == "LeoJiangOR/vitra-gigahands-keypoints-step140000"
    assert weights_path.name == "epoch=0-step=140000.ckpt__weights.pt"
    assert config_path.name == "config__human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json"


def test_resolve_local_checkpoint_and_default_config(tmp_path):
    ckpt = tmp_path / "weights.pt"
    ckpt.write_text("not really torch", encoding="utf-8")
    weights_path, config_path, repo_id = resolve_checkpoint_and_config(str(ckpt))
    assert weights_path == ckpt
    assert str(config_path).endswith("human_pretrain_gigahands_real_full_keypoints_vitra3b_linked.json")
    assert repo_id is None


def test_seconds_to_chunk_index_uses_fps_and_clamps():
    assert seconds_to_chunk_index(0.33, fps=8, chunk_len=16) == 3
    assert seconds_to_chunk_index(0.66, fps=8, chunk_len=16) == 5
    assert seconds_to_chunk_index(100.0, fps=8, chunk_len=16) == 16


def test_touch_eval_metrics_use_editable_mask_and_hand_splits():
    totals = {}
    diff = torch.zeros((1, 2, 4), dtype=torch.float32)
    diff[..., 0] = 1.0
    diff[..., 2] = 3.0
    editable = torch.tensor([[[True, False, True, False], [False, False, True, False]]])

    add_hand_mse(totals, "edit_1_base", diff, editable)
    add_hand_mse(totals, "edit_1_edit", diff * 0.5, editable)
    add_delta_stats(totals, "edit_1", diff, editable.to(torch.float32))
    totals["edit_1_valid_editable_dims"] = float(editable.sum())

    metrics = finalize_metrics(totals, edit_count=1)

    assert metrics["edit_1_base_mse"] == pytest.approx((1.0 + 9.0 + 9.0) / 3.0)
    assert metrics["edit_1_edit_mse"] == pytest.approx(metrics["edit_1_base_mse"] * 0.25)
    assert metrics["edit_1_improvement_pct"] == pytest.approx(75.0)
    assert metrics["edit_1_left_base_mse"] == pytest.approx(1.0)
    assert metrics["edit_1_right_base_mse"] == pytest.approx(9.0)
    assert metrics["edit_1_valid_editable_dims"] == 3.0


def test_touch_eval_ablation_helpers_shuffle_touch_and_targets():
    batch = {
        "touch_pressure": torch.arange(2 * 3 * 2 * 16 * 16, dtype=torch.float32).reshape(2, 3, 2, 16, 16),
        "touch_mask": torch.tensor([[[True, False], [True, False], [True, False]], [[False, True], [False, True], [False, True]]]),
        "a_target": torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
        "action_mask": torch.tensor([[[True, False]], [[False, True]]]),
    }

    shuffled_pressure, shuffled_mask = shuffle_touch(batch)
    assert torch.allclose(shuffled_pressure[0], batch["touch_pressure"][1])
    assert torch.equal(shuffled_mask[0], batch["touch_mask"][1])

    randomize_targets_from_batch(batch)
    assert torch.allclose(batch["a_target"][0], torch.tensor([[3.0, 4.0]]))
    assert torch.equal(batch["action_mask"][0], torch.tensor([[False, True]]))


@pytest.mark.parametrize("ablation", ["matched", "zero_touch", "shuffled_touch", "no_touch"])
def test_touch_eval_accepts_design1_ablation_modes(monkeypatch, ablation):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_touch_guided_actions.py",
            "--cache_root",
            "cache",
            "--touch_editor_checkpoint",
            "editor.pt",
            "--ablation",
            ablation,
        ],
    )

    args = parse_eval_args()

    assert args.ablation == ablation


def test_no_touch_training_ablation_zeroes_pressure_and_mask():
    pressure = torch.ones((2, 3, 2, 16, 16), dtype=torch.float32)
    mask = torch.ones((2, 3, 2), dtype=torch.bool)

    ablated_pressure, ablated_mask = apply_touch_training_ablation(pressure, mask, "zero_touch")
    kept_pressure, kept_mask = apply_touch_training_ablation(pressure, mask, "none")

    assert torch.count_nonzero(ablated_pressure) == 0
    assert not ablated_mask.any()
    assert torch.equal(kept_pressure, pressure)
    assert torch.equal(kept_mask, mask)


def test_high_contact_subset_uses_top_quartile_touch_pressure(tmp_path):
    for idx, value in enumerate([1.0, 2.0, 3.0, 4.0]):
        path = tmp_path / f"sample_{idx:08d}.npz"
        write_cache_sample(path)
        payload = dict(np.load(path, allow_pickle=False))
        payload["touch_pressure"] = np.full((16, 2, 16, 16), value, dtype=np.float32)
        payload["touch_mask"] = np.ones((16, 2), dtype=bool)
        np.savez(path, **payload)

    ds = TouchEditorCacheDataset(tmp_path)
    selected, threshold = high_contact_path_set(ds, quantile=0.75)

    assert threshold == pytest.approx(3.25)
    assert len(selected) == 1
    assert next(iter(selected)).endswith("sample_00000003.npz")
    assert touch_contact_score(ds[3]["touch_pressure"], ds[3]["touch_mask"]) == pytest.approx(4.0)


def test_filter_batch_by_paths_keeps_selected_samples():
    batch = {
        "path": ["a.npz", "b.npz"],
        "a_base": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "scalar": torch.tensor(1.0),
    }

    filtered = filter_batch_by_paths(batch, {"b.npz"})

    assert filtered is not None
    assert filtered["path"] == ["b.npz"]
    assert torch.equal(filtered["a_base"], torch.tensor([[2.0, 3.0]]))
    assert torch.equal(filtered["scalar"], torch.tensor(1.0))


def test_touch_specific_gain_positive_when_matched_beats_baseline():
    abs_gain, pct_gain = gain(matched=0.2, baseline=0.5)

    assert abs_gain == pytest.approx(0.3)
    assert pct_gain == pytest.approx(60.0)


class ConstantDeltaEditor(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.value = value

    def forward(
        self,
        a_base,
        current_state,
        current_state_mask,
        touch_pressure,
        touch_mask,
        chunk_phase,
        future_mask,
        action_mask,
    ):
        return torch.ones_like(a_base) * self.value


def test_apply_touch_editor_once_edits_only_unexecuted_future():
    editor = ConstantDeltaEditor(2.0)
    a_base = torch.zeros((1, 16, 192), dtype=torch.float32)
    action_mask = torch.ones((1, 16, 192), dtype=torch.bool)
    action_mask[:, :, 100:] = False
    a_edit, delta, future_mask = apply_touch_editor_once(
        editor=editor,
        a_base=a_base,
        current_state=torch.zeros((1, 212), dtype=torch.float32),
        current_state_mask=torch.ones((1, 212), dtype=torch.bool),
        touch_pressure=torch.ones((1, 16, 2, 16, 16), dtype=torch.float32),
        touch_mask=torch.ones((1, 16, 2), dtype=torch.bool),
        action_mask=action_mask,
        edit_start_idx=5,
    )
    assert delta.shape == a_base.shape
    assert future_mask[:, :5].sum() == 0
    assert torch.allclose(a_edit[:, :5], a_base[:, :5])
    assert torch.allclose(a_edit[:, 5:, :100], torch.full((1, 11, 100), 2.0))
    assert torch.allclose(a_edit[:, 5:, 100:], torch.zeros((1, 11, 92)))


def test_apply_touch_guidance_schedule_is_sequential():
    editor = ConstantDeltaEditor(1.0)
    a_base = torch.zeros((1, 16, 192), dtype=torch.float32)
    result = apply_touch_guidance_schedule(
        editor=editor,
        a_base=a_base,
        current_state=torch.zeros((1, 212), dtype=torch.float32),
        current_state_mask=torch.ones((1, 212), dtype=torch.bool),
        touch_pressure=torch.ones((1, 16, 2, 16, 16), dtype=torch.float32),
        touch_mask=torch.ones((1, 16, 2), dtype=torch.bool),
        action_mask=torch.ones((1, 16, 192), dtype=torch.bool),
        fps=8,
        edit_times=(0.33, 0.66),
    )
    assert result.edit_indices == [3, 5]
    assert len(result.a_history) == 2
    assert torch.allclose(result.a_edit[:, :3], torch.zeros((1, 3, 192)))
    assert torch.allclose(result.a_edit[:, 3:5], torch.ones((1, 2, 192)))
    assert torch.allclose(result.a_edit[:, 5:], torch.full((1, 11, 192), 2.0))


def test_apply_touch_guidance_broadcasts_state_touch_and_mask_for_multiple_samples():
    editor = ConstantDeltaEditor(1.0)
    result = apply_touch_guidance_schedule(
        editor=editor,
        a_base=torch.zeros((2, 16, 192), dtype=torch.float32),
        current_state=torch.zeros((1, 212), dtype=torch.float32),
        current_state_mask=torch.ones((1, 212), dtype=torch.bool),
        touch_pressure=torch.ones((1, 16, 2, 16, 16), dtype=torch.float32),
        touch_mask=torch.ones((1, 16, 2), dtype=torch.bool),
        action_mask=torch.ones((1, 16, 192), dtype=torch.bool),
        fps=8,
        edit_times=(0.33,),
    )
    assert result.a_edit.shape == (2, 16, 192)
    assert torch.allclose(result.a_edit[:, :3], torch.zeros((2, 3, 192)))
    assert torch.allclose(result.a_edit[:, 3:], torch.ones((2, 13, 192)))
