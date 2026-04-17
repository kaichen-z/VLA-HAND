import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = REPO_ROOT / "tools" / "prepare_gigahands_real_subset.py"
CONVERTER_PATH = REPO_ROOT / "data" / "preprocessing" / "convert_gigahands_to_vitra_stage1.py"
EVAL_PATH = REPO_ROOT / "tools" / "evaluate_gigahands_stage1.py"
VERIFY_VIDEOS_PATH = REPO_ROOT / "tools" / "verify_required_videos.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_params(path: Path, frame_count: int = 40, right_offset: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    poses = np.zeros((frame_count, 48), dtype=np.float32)
    poses[:, 3] = np.linspace(0.0, 0.2, frame_count)
    left_th = np.stack([np.linspace(0.0, 0.3, frame_count), np.zeros(frame_count), np.ones(frame_count)], axis=1)
    right_th = left_th + np.array([right_offset, 0.0, 0.0], dtype=np.float32)
    payload = {
        "left": {
            "poses": poses.tolist(),
            "Rh": np.zeros((frame_count, 3), dtype=np.float32).tolist(),
            "Th": left_th.astype(np.float32).tolist(),
            "shapes": np.zeros((frame_count, 10), dtype=np.float32).tolist(),
        },
        "right": {
            "poses": poses.tolist(),
            "Rh": np.zeros((frame_count, 3), dtype=np.float32).tolist(),
            "Th": right_th.astype(np.float32).tolist(),
            "shapes": np.zeros((frame_count, 10), dtype=np.float32).tolist(),
        },
    }
    path.write_text(json.dumps(payload))


def write_camera(path: Path, camera_name: str = "brics-odroid-011_cam0") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = [
        11,
        1280,
        720,
        900.0,
        900.0,
        640.0,
        360.0,
        0.0,
        0.0,
        0.0,
        0.0,
        camera_name,
        1.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.2,
        0.3,
    ]
    path.write_text(" ".join(map(str, row)) + "\n")


def write_full_fixture(root: Path, count: int = 4) -> None:
    scene = "p001-box"
    sequence_root = root / "hand_poses" / scene
    write_camera(sequence_root / "optim_params.txt")
    (root / "multiview_rgb_vids" / scene / "brics-odroid-011_cam0").mkdir(parents=True, exist_ok=True)
    rows = []
    map_rows = ["scene,sequence,brics-odroid-011_cam0"]
    for idx in range(count):
        sequence = f"{idx + 1:03d}"
        write_params(sequence_root / "params" / f"{sequence}.json")
        video_rel = f"multiview_rgb_vids/{scene}/brics-odroid-011_cam0/brics-odroid-011_cam0_{sequence}.mp4"
        (root / video_rel).write_bytes(b"fake mp4")
        rows.append(
            {
                "scene": scene,
                "sequence": [sequence],
                "start_frame_id": 0,
                "end_frame_id": 35,
                "clarify_annotation": f"Pick up object {idx}.",
                "rewritten_annotation": [f"Pick up object {idx} with both hands."],
            }
        )
        map_rows.append(f"{scene},{idx + 1},brics-odroid-011_cam0/brics-odroid-011_cam0_171814{idx}.mp4")
    (root / "annotations_v2.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (root / "multiview_camera_video_map.csv").write_text("\n".join(map_rows) + "\n")


def test_prepare_real_subset_writes_manifest_and_needed_videos(tmp_path):
    module = load_module(PREPARE_PATH, "prepare_gigahands_real_subset")
    root = tmp_path / "GigaHands_subset_real"
    write_full_fixture(root, count=4)

    manifest = tmp_path / "subset_manifest.json"
    needed_videos = tmp_path / "needed_videos.txt"
    report = module.prepare_real_subset(
        gigahands_root=root,
        num_train=2,
        num_test=1,
        min_frames=32,
        prefer_camera="brics-odroid-011_cam0",
        require_both_hands_valid=True,
        prefer_bimanual_motion=True,
        candidate_pool_factor=4,
        output_manifest=manifest,
        output_video_list=needed_videos,
    )

    payload = json.loads(manifest.read_text())
    assert report["selected_train"] == 2
    assert report["selected_test"] == 1
    assert len(payload["splits"]["train"]) == 2
    assert len(payload["splits"]["test"]) == 1
    assert payload["clips"][0]["camera"] == "brics-odroid-011_cam0"
    assert payload["clips"][0]["instruction"].endswith(".")
    assert payload["clips"][0]["video_path"].startswith("multiview_rgb_vids/p001-box/brics-odroid-011_cam0/")
    assert "171814" in payload["clips"][0]["video_path"]
    assert needed_videos.read_text().count(".mp4") == 3


def test_prepare_real_subset_requires_hand_poses(tmp_path):
    module = load_module(PREPARE_PATH, "prepare_gigahands_real_subset")
    root = tmp_path / "GigaHands_subset_real"
    root.mkdir()
    (root / "annotations_v2.jsonl").write_text(
        json.dumps(
            {
                "scene": "p001-folder",
                "sequence": "000",
                "start_frame_id": 0,
                "end_frame_id": -1,
                "rewritten_annotation": ["Hold the object."],
            }
        )
        + "\n"
    )

    with pytest.raises(FileNotFoundError, match="hand_poses"):
        module.prepare_real_subset(
            gigahands_root=root,
            num_train=1,
            num_test=0,
            min_frames=32,
            prefer_camera="brics-odroid-011_cam0",
            require_both_hands_valid=True,
            prefer_bimanual_motion=True,
            candidate_pool_factor=4,
            output_manifest=tmp_path / "subset_manifest.json",
            output_video_list=tmp_path / "needed_videos.txt",
        )


def test_prepare_real_subset_can_stop_after_small_candidate_pool(tmp_path):
    module = load_module(PREPARE_PATH, "prepare_gigahands_real_subset")
    root = tmp_path / "GigaHands_subset_real"
    write_full_fixture(root, count=5)

    report = module.prepare_real_subset(
        gigahands_root=root,
        num_train=1,
        num_test=1,
        min_frames=32,
        prefer_camera="brics-odroid-011_cam0",
        require_both_hands_valid=True,
        prefer_bimanual_motion=True,
        candidate_pool_factor=1,
        output_manifest=tmp_path / "subset_manifest.json",
        output_video_list=tmp_path / "needed_videos.txt",
    )

    assert report["candidates"] == 2
    assert report["selected_train"] == 1
    assert report["selected_test"] == 1


def test_video_map_loader_falls_back_to_comma_csv_when_sniffer_fails(tmp_path, monkeypatch):
    module = load_module(PREPARE_PATH, "prepare_gigahands_real_subset")
    map_path = tmp_path / "multiview_camera_video_map.csv"
    map_path.write_text(
        "scene,sequence,brics-odroid-011_cam0\n"
        "p001-folder,0,brics-odroid-011_cam0/brics-odroid-011_cam0_1718141283460559.mp4\n"
    )

    class BrokenSniffer:
        def sniff(self, sample):
            raise module.csv.Error("Could not determine delimiter")

    monkeypatch.setattr(module.csv, "Sniffer", BrokenSniffer)

    video_map = module.load_video_map(map_path)

    assert ("p001-folder", "000") in video_map
    assert video_map[("p001-folder", "000")][0]["brics-odroid-011_cam0"].endswith(".mp4")


def test_converter_full_manifest_writes_train_and_test_datasets(tmp_path):
    converter = load_module(CONVERTER_PATH, "convert_gigahands_to_vitra_stage1")
    root = tmp_path / "GigaHands_subset_real"
    output_root = tmp_path / "vitra_gigahands_real_subset"
    write_full_fixture(root, count=3)
    manifest = {
        "clips": [
            {
                "clip_id": "train_001",
                "split": "train",
                "scene": "p001-box",
                "sequence_id": "001",
                "camera": "brics-odroid-011_cam0",
                "instruction": "Pick up object 0 with both hands.",
                "start_frame": 0,
                "end_frame": 36,
                "params_path": "hand_poses/p001-box/params/001.json",
                "camera_path": "hand_poses/p001-box/optim_params.txt",
                "video_path": "multiview_rgb_vids/p001-box/brics-odroid-011_cam0/brics-odroid-011_cam0_001.mp4",
            },
            {
                "clip_id": "test_002",
                "split": "test",
                "scene": "p001-box",
                "sequence_id": "002",
                "camera": "brics-odroid-011_cam0",
                "instruction": "Pick up object 1 with both hands.",
                "start_frame": 0,
                "end_frame": 36,
                "params_path": "hand_poses/p001-box/params/002.json",
                "camera_path": "hand_poses/p001-box/optim_params.txt",
                "video_path": "multiview_rgb_vids/p001-box/brics-odroid-011_cam0/brics-odroid-011_cam0_002.mp4",
            },
        ],
        "splits": {"train": ["train_001"], "test": ["test_002"]},
    }
    manifest_path = root / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    report = converter.convert_gigahands_to_vitra(
        gigahands_root=root,
        output_root=output_root,
        input_layout="full",
        subset_manifest=manifest_path,
        split="all",
        camera="auto",
        dataset_name_prefix="gigahands_real",
        min_frames=17,
        min_valid_ratio=0.9,
        write_video=True,
        undistort=False,
    )

    assert report["num_episodes_written"] == 2
    train_index = output_root / "Annotation" / "gigahands_real_train" / "episode_frame_index.npz"
    test_index = output_root / "Annotation" / "gigahands_real_test" / "episode_frame_index.npz"
    assert train_index.exists()
    assert test_index.exists()
    train_episodes = list((output_root / "Annotation" / "gigahands_real_train" / "episodic_annotations").glob("*.npy"))
    test_episodes = list((output_root / "Annotation" / "gigahands_real_test" / "episodic_annotations").glob("*.npy"))
    assert len(train_episodes) == 1
    assert len(test_episodes) == 1
    epi = np.load(train_episodes[0], allow_pickle=True).item()
    assert epi["video_name"].startswith("p001-box/brics-odroid-011_cam0/")
    assert epi["text"]["left"][0][0] == "Pick up object 0 with both hands."


def test_converted_real_train_split_can_be_read_by_frame_dataset_without_images(tmp_path):
    converter = load_module(CONVERTER_PATH, "convert_gigahands_to_vitra_stage1")
    root = tmp_path / "GigaHands_subset_real"
    output_root = tmp_path / "vitra_gigahands_real_subset"
    write_full_fixture(root, count=2)
    manifest = {
        "clips": [
            {
                "clip_id": "train_001",
                "split": "train",
                "scene": "p001-box",
                "sequence_id": "001",
                "camera": "brics-odroid-011_cam0",
                "instruction": "Pick up object 0 with both hands.",
                "start_frame": 0,
                "end_frame": 36,
                "params_path": "hand_poses/p001-box/params/001.json",
                "camera_path": "hand_poses/p001-box/optim_params.txt",
                "video_path": "multiview_rgb_vids/p001-box/brics-odroid-011_cam0/brics-odroid-011_cam0_001.mp4",
            }
        ],
        "splits": {"train": ["train_001"], "test": []},
    }
    manifest_path = root / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    converter.convert_gigahands_to_vitra(
        gigahands_root=root,
        output_root=output_root,
        input_layout="full",
        subset_manifest=manifest_path,
        split="all",
        camera="auto",
        dataset_name_prefix="gigahands_real",
        min_frames=17,
        min_valid_ratio=0.9,
        write_video=True,
        undistort=False,
    )

    from vitra.datasets.dataset import FrameDataset

    ds = FrameDataset(
        dataset_folder=str(output_root),
        dataset_name="gigahands_real_train",
        action_future_window_size=16,
        augmentation=False,
        load_images=False,
        state_mask_prob=0.0,
    )
    item = ds[0]

    assert len(ds) == 36
    assert item["action_list"].shape == (17, 102)
    assert item["current_state"].shape == (122,)
    assert item["action_mask"].shape == (17, 2)


def test_real_gigahands_dataset_registration_is_present():
    mixtures_py = (REPO_ROOT / "vitra" / "datasets" / "data_mixture.py").read_text()
    dataset_py = (REPO_ROOT / "vitra" / "datasets" / "dataset.py").read_text()
    stats_py = (REPO_ROOT / "vitra" / "datasets" / "calculate_statistics.py").read_text()

    assert '"gigahands_real_train"' in mixtures_py
    assert '"gigahands_real_test"' in mixtures_py
    assert "dataset_name.startswith('gigahands_real_')" in dataset_py
    assert "gigahands_real_train" in stats_py


def test_evaluator_metric_helpers_compute_masked_errors():
    module = load_module(EVAL_PATH, "evaluate_gigahands_stage1")
    target = np.ones((2, 4, 192), dtype=np.float32)
    prediction = target.copy()
    prediction[:, :, :96] += 1.0
    action_masks = np.zeros((2, 4, 2), dtype=bool)
    action_masks[:, :, 0] = True

    metrics = module.compute_action_metrics(prediction, target, action_masks)

    assert metrics["valid_frame_count"] == 8
    assert metrics["left_action_mse"] == 1.0
    assert metrics["right_action_mse"] == 0.0
    assert metrics["dual_hand_action_mse"] == 0.0


def test_video_requirement_verifier_reports_missing_files(tmp_path):
    module = load_module(VERIFY_VIDEOS_PATH, "verify_required_videos")
    root = tmp_path / "GigaHands_subset_real"
    existing = root / "multiview_rgb_vids" / "p001-box" / "cam0" / "clip_001.mp4"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"fake mp4")
    required = root / "needed_videos.txt"
    required.write_text(
        "multiview_rgb_vids/p001-box/cam0/clip_001.mp4\n"
        "multiview_rgb_vids/p001-box/cam0/clip_001.mp4\n"
        "multiview_rgb_vids/p001-box/cam0/clip_002.mp4\n"
    )

    report = module.verify_required_videos(root, required, unique_output=root / "needed_videos_unique.txt")

    assert report["listed"] == 3
    assert report["needed"] == 2
    assert report["present"] == 1
    assert report["missing"] == 1
    assert report["missing_paths"] == ["multiview_rgb_vids/p001-box/cam0/clip_002.mp4"]
    assert (root / "needed_videos_unique.txt").read_text().splitlines() == [
        "multiview_rgb_vids/p001-box/cam0/clip_001.mp4",
        "multiview_rgb_vids/p001-box/cam0/clip_002.mp4",
    ]


def test_pipeline_scripts_are_portable_and_expose_repo_stages():
    gigahands_script = (REPO_ROOT / "scripts" / "run_gigahands_real_subset_pipeline.sh").read_text()
    opentouch_script = (REPO_ROOT / "scripts" / "run_opentouch_stage1_subset_pipeline.sh").read_text()
    gigahands_config = (REPO_ROOT / "vitra" / "configs" / "human_pretrain_gigahands_real_subset.json").read_text()
    opentouch_config = (REPO_ROOT / "vitra" / "configs" / "human_pretrain_opentouch_keypoint_subset.json").read_text()

    for payload in [gigahands_script, opentouch_script, gigahands_config, opentouch_config]:
        assert "/home/chonghej" not in payload
    assert "verify_videos)" in gigahands_script
    assert "make_unique_video_list)" in gigahands_script
    assert "download_first_available" in opentouch_script
    assert "verify_raw)" in opentouch_script
    assert "eval_before)" in opentouch_script
    assert "eval_after)" in opentouch_script


def test_evaluator_dataset_kwargs_preserve_configured_action_representation():
    module = load_module(EVAL_PATH, "evaluate_gigahands_stage1")
    config = {
        "train_dataset": {
            "action_type": "keypoints",
            "use_rel": False,
            "rel_mode": "step",
            "clip_len": None,
            "state_mask_prob": 0.0,
            "target_image_height": 224,
        }
    }

    kwargs = module.dataset_kwargs_from_config(config)

    assert kwargs["action_type"] == "keypoints"
    assert kwargs["use_rel"] is False
    assert kwargs["rel_mode"] == "step"
    assert kwargs["clip_len"] is None
    assert kwargs["state_mask_prob"] == 0.0
