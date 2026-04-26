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


def write_single_valid_hand_params(path: Path, frame_count: int = 40, valid_side: str = "left") -> None:
    write_params(path, frame_count=frame_count)
    payload = json.loads(path.read_text())
    invalid_side = "right" if valid_side == "left" else "left"
    payload[invalid_side]["Th"] = (np.full((frame_count, 3), np.nan, dtype=np.float32)).tolist()
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


def write_cameras(path: Path, camera_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, camera_name in enumerate(camera_names):
        row = [
            idx + 1,
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
        rows.append(" ".join(map(str, row)))
    path.write_text("\n".join(rows) + "\n")


def write_keypoints(
    path: Path,
    frame_count: int = 40,
    *,
    left_offset: float = 0.0,
    right_offset: float = 0.25,
    invalid_frame: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.arange(frame_count, dtype=np.float32)[:, None, None]
    joint_ids = np.arange(21, dtype=np.float32)[None, :, None]
    xyz = np.array([1.0, 0.5, 0.25], dtype=np.float32)[None, None, :]
    left = left_offset + frames * 0.01 + joint_ids * 0.001 + xyz
    right = right_offset + frames * 0.02 + joint_ids * 0.002 + xyz
    if invalid_frame is not None:
        left[invalid_frame] = np.nan
    payload = {
        "left": left.reshape(frame_count, -1).tolist(),
        "right": right.reshape(frame_count, -1).tolist(),
    }
    path.write_text(json.dumps(payload))


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


def write_multiview_fixture(root: Path, count: int = 4) -> None:
    scene = "p001-box"
    cameras = ["brics-odroid-001_cam0", "brics-odroid-001_cam1", "brics-odroid-002_cam0"]
    sequence_root = root / "hand_poses" / scene
    write_cameras(sequence_root / "optim_params.txt", cameras)
    rows = []
    header = ["scene", "sequence", *cameras]
    map_rows = [",".join(header)]
    for idx in range(count):
        sequence = f"{idx + 1:03d}"
        write_params(sequence_root / "params" / f"{sequence}.json")
        write_keypoints(sequence_root / "keypoints_3d_mano" / f"{sequence}.json", left_offset=100.0 + idx)
        values = [scene, str(idx + 1)]
        for camera in cameras:
            video_rel = f"multiview_rgb_vids/{scene}/{camera}/{camera}_{sequence}.mp4"
            (root / video_rel).parent.mkdir(parents=True, exist_ok=True)
            (root / video_rel).write_bytes(b"fake mp4")
            values.append(f"{camera}/{camera}_{sequence}.mp4")
        map_rows.append(",".join(values))
        rows.append(
            {
                "scene": scene,
                "sequence": [sequence],
                "start_frame_id": 0,
                "end_frame_id": 35,
                "clarify_annotation": f"Pick up object {idx}.",
                "rewritten_annotation": [f"Pick up object {idx}."],
            }
        )
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
        require_keypoints=False,
        require_real_keypoints=False,
        prefer_bimanual_motion=True,
        candidate_pool_factor=4,
        require_video_exists=False,
        require_video_frame_count=False,
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
            require_keypoints=False,
            require_real_keypoints=False,
            prefer_bimanual_motion=True,
            candidate_pool_factor=4,
            require_video_exists=False,
            require_video_frame_count=False,
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
        require_keypoints=False,
        require_real_keypoints=False,
        prefer_bimanual_motion=True,
        candidate_pool_factor=1,
        require_video_exists=False,
        require_video_frame_count=False,
        output_manifest=tmp_path / "subset_manifest.json",
        output_video_list=tmp_path / "needed_videos.txt",
    )

    assert report["candidates"] == 2
    assert report["selected_train"] == 1
    assert report["selected_test"] == 1


def test_prepare_real_subset_can_require_real_keypoints(tmp_path):
    module = load_module(PREPARE_PATH, "prepare_gigahands_real_subset")
    root = tmp_path / "GigaHands_subset_real"
    write_full_fixture(root, count=3)
    scene_root = root / "hand_poses" / "p001-box"
    write_keypoints(scene_root / "keypoints_3d_mano" / "001.json")
    write_keypoints(scene_root / "keypoints_3d" / "002.json")
    write_keypoints(scene_root / "keypoints_3d_mano" / "003.json")

    report = module.prepare_real_subset(
        gigahands_root=root,
        num_train=1,
        num_test=0,
        min_frames=32,
        prefer_camera="brics-odroid-011_cam0",
        require_both_hands_valid=True,
        require_keypoints=True,
        require_real_keypoints=True,
        prefer_bimanual_motion=True,
        candidate_pool_factor=4,
        require_video_exists=False,
        require_video_frame_count=False,
        output_manifest=tmp_path / "subset_manifest.json",
        output_video_list=tmp_path / "needed_videos.txt",
    )

    manifest = json.loads((tmp_path / "subset_manifest.json").read_text())
    assert report["candidates"] == 1
    assert manifest["clips"][0]["sequence_id"] == "002"


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


def test_prepare_select_all_expands_only_cam0_views_and_groups_split(tmp_path):
    module = load_module(PREPARE_PATH, "prepare_gigahands_real_subset")
    root = tmp_path / "GigaHands_subset_real"
    write_multiview_fixture(root, count=4)

    manifest = tmp_path / "subset_manifest.json"
    needed_videos = tmp_path / "needed_videos.txt"
    report = module.prepare_real_subset(
        gigahands_root=root,
        num_train=0,
        num_test=0,
        min_frames=32,
        prefer_camera="brics-odroid-001_cam0",
        require_both_hands_valid=False,
        require_keypoints=True,
        require_real_keypoints=False,
        keypoints_source="mano",
        prefer_bimanual_motion=False,
        candidate_pool_factor=4,
        require_video_exists=True,
        require_video_frame_count=False,
        output_manifest=manifest,
        output_video_list=needed_videos,
        select_all=True,
        test_ratio=0.25,
        camera_scope="all_cam0",
        seed=7,
    )

    payload = json.loads(manifest.read_text())
    cameras = {clip["camera"] for clip in payload["clips"]}
    base_to_splits = {}
    for clip in payload["clips"]:
        base_key = (clip["scene"], clip["sequence_id"], clip["start_frame"], clip["end_frame"])
        base_to_splits.setdefault(base_key, set()).add(clip["split"])

    assert report["selected_train"] == 6
    assert report["selected_test"] == 2
    assert cameras == {"brics-odroid-001_cam0", "brics-odroid-002_cam0"}
    assert all(not clip["camera"].endswith("_cam1") for clip in payload["clips"])
    assert all(len(splits) == 1 for splits in base_to_splits.values())
    assert needed_videos.read_text().count(".mp4") == 8


def test_prepare_keypoints_source_mano_requires_mano_keypoints(tmp_path):
    module = load_module(PREPARE_PATH, "prepare_gigahands_real_subset")
    root = tmp_path / "GigaHands_subset_real"
    write_full_fixture(root, count=2)
    scene_root = root / "hand_poses" / "p001-box"
    write_keypoints(scene_root / "keypoints_3d" / "001.json")
    write_keypoints(scene_root / "keypoints_3d_mano" / "002.json")

    report = module.prepare_real_subset(
        gigahands_root=root,
        num_train=1,
        num_test=0,
        min_frames=32,
        prefer_camera="brics-odroid-011_cam0",
        require_both_hands_valid=True,
        require_keypoints=True,
        require_real_keypoints=False,
        keypoints_source="mano",
        prefer_bimanual_motion=True,
        candidate_pool_factor=4,
        require_video_exists=False,
        require_video_frame_count=False,
        output_manifest=tmp_path / "subset_manifest.json",
        output_video_list=tmp_path / "needed_videos.txt",
    )

    manifest = json.loads((tmp_path / "subset_manifest.json").read_text())
    assert report["candidates"] == 1
    assert manifest["clips"][0]["sequence_id"] == "002"


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
    assert epi["camera"]["name"] == "brics-odroid-011_cam0"
    assert epi["camera"]["image_size"] == [1280, 720]
    assert epi["camera"]["undistorted"] is False
    np.testing.assert_allclose(epi["camera"]["intrinsics"], epi["intrinsics"])
    np.testing.assert_allclose(epi["camera"]["distortion"], np.zeros(4, dtype=np.float32))
    assert report["undistorted_requested"] is False
    assert report["num_raw_copied_videos"] == 2
    assert report["num_undistorted_videos"] == 0


def test_converter_prefers_real_keypoints_over_mano_keypoints(tmp_path):
    converter = load_module(CONVERTER_PATH, "convert_gigahands_to_vitra_stage1")
    root = tmp_path / "GigaHands_subset_real"
    output_root = tmp_path / "vitra_gigahands_real_subset"
    write_full_fixture(root, count=1)
    scene_root = root / "hand_poses" / "p001-box"
    write_keypoints(scene_root / "keypoints_3d" / "001.json", left_offset=10.0, right_offset=20.0)
    write_keypoints(scene_root / "keypoints_3d_mano" / "001.json", left_offset=100.0, right_offset=200.0)

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
        write_video=False,
    )

    train_episodes = list((output_root / "Annotation" / "gigahands_real_train" / "episodic_annotations").glob("*.npy"))
    epi = np.load(train_episodes[0], allow_pickle=True).item()
    assert report["used_keypoint_fallback_joints"] == 1
    np.testing.assert_allclose(epi["left"]["joints_worldspace"][0, 0], np.array([11.0, 10.5, 10.25], dtype=np.float32))
    np.testing.assert_allclose(epi["right"]["joints_worldspace"][0, 0], np.array([21.0, 20.5, 20.25], dtype=np.float32))


def test_converter_can_require_real_keypoints_and_reject_mano_only(tmp_path):
    converter = load_module(CONVERTER_PATH, "convert_gigahands_to_vitra_stage1")
    root = tmp_path / "GigaHands_subset_real"
    output_root = tmp_path / "vitra_gigahands_real_subset"
    write_full_fixture(root, count=1)
    scene_root = root / "hand_poses" / "p001-box"
    write_keypoints(scene_root / "keypoints_3d_mano" / "001.json", left_offset=100.0, right_offset=200.0)

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
        write_video=False,
        require_real_keypoints=True,
    )

    assert report["num_episodes_written"] == 0
    assert report["skipped_missing_real_keypoints"] == 1


def test_converter_can_force_mano_keypoints_over_real_keypoints(tmp_path):
    converter = load_module(CONVERTER_PATH, "convert_gigahands_to_vitra_stage1")
    root = tmp_path / "GigaHands_subset_real"
    output_root = tmp_path / "vitra_gigahands_real_subset"
    write_full_fixture(root, count=1)
    scene_root = root / "hand_poses" / "p001-box"
    write_keypoints(scene_root / "keypoints_3d" / "001.json", left_offset=10.0, right_offset=20.0)
    write_keypoints(scene_root / "keypoints_3d_mano" / "001.json", left_offset=100.0, right_offset=200.0)

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
        write_video=False,
        keypoints_source="mano",
    )

    train_episodes = list((output_root / "Annotation" / "gigahands_real_train" / "episodic_annotations").glob("*.npy"))
    epi = np.load(train_episodes[0], allow_pickle=True).item()
    assert report["used_mano_keypoints"] == 1
    assert report["used_real_keypoints"] == 0
    np.testing.assert_allclose(epi["left"]["joints_worldspace"][0, 0], np.array([101.0, 100.5, 100.25], dtype=np.float32))
    np.testing.assert_allclose(epi["right"]["joints_worldspace"][0, 0], np.array([201.0, 200.5, 200.25], dtype=np.float32))


def test_converter_filters_invalid_real_keypoint_frames_from_kept_mask(tmp_path):
    converter = load_module(CONVERTER_PATH, "convert_gigahands_to_vitra_stage1")
    root = tmp_path / "GigaHands_subset_real"
    output_root = tmp_path / "vitra_gigahands_real_subset"
    write_full_fixture(root, count=1)
    scene_root = root / "hand_poses" / "p001-box"
    write_keypoints(scene_root / "keypoints_3d" / "001.json", invalid_frame=5)

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
        write_video=False,
        require_real_keypoints=True,
    )

    train_episodes = list((output_root / "Annotation" / "gigahands_real_train" / "episodic_annotations").glob("*.npy"))
    epi = np.load(train_episodes[0], allow_pickle=True).item()
    assert bool(epi["left"]["kept_frames"][5]) is False
    assert bool(epi["right"]["kept_frames"][5]) is True


def test_converter_keeps_single_hand_valid_clips_with_masks(tmp_path):
    converter = load_module(CONVERTER_PATH, "convert_gigahands_to_vitra_stage1")
    root = tmp_path / "GigaHands_subset_real"
    output_root = tmp_path / "vitra_gigahands_real_subset"
    write_full_fixture(root, count=1)
    scene_root = root / "hand_poses" / "p001-box"
    write_single_valid_hand_params(scene_root / "params" / "001.json", valid_side="left")
    write_keypoints(scene_root / "keypoints_3d_mano" / "001.json", left_offset=100.0, right_offset=200.0)

    manifest = {
        "clips": [
            {
                "clip_id": "train_001",
                "split": "train",
                "scene": "p001-box",
                "sequence_id": "001",
                "camera": "brics-odroid-011_cam0",
                "instruction": "Move the left hand.",
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
        write_video=False,
        keypoints_source="mano",
    )

    train_episodes = list((output_root / "Annotation" / "gigahands_real_train" / "episodic_annotations").glob("*.npy"))
    epi = np.load(train_episodes[0], allow_pickle=True).item()
    assert report["num_episodes_written"] == 1
    assert np.mean(epi["left"]["kept_frames"]) == 1.0
    assert np.mean(epi["right"]["kept_frames"]) == 0.0


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


def test_converter_clean_output_removes_previous_managed_files(tmp_path):
    converter = load_module(CONVERTER_PATH, "convert_gigahands_to_vitra_stage1")
    root = tmp_path / "GigaHands_subset_real"
    output_root = tmp_path / "vitra_gigahands_real_subset"
    write_full_fixture(root, count=1)
    manifest = {
        "clips": [
            {
                "clip_id": "train_001",
                "split": "train",
                "scene": "p001-box",
                "sequence_id": "001",
                "camera": "brics-odroid-011_cam0",
                "instruction": "Pick up object with both hands.",
                "start_frame": 0,
                "end_frame": 36,
                "params_path": "hand_poses/p001-box/params/001.json",
                "camera_path": "hand_poses/p001-box/optim_params.txt",
                "video_path": "multiview_rgb_vids/p001-box/brics-odroid-011_cam0/brics-odroid-011_cam0_001.mp4",
            }
        ],
    }
    manifest_path = root / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    orphan_annotation = output_root / "Annotation" / "old" / "stale.npy"
    orphan_video = output_root / "Video" / "GigaHands_root" / "old" / "stale.mp4"
    orphan_annotation.parent.mkdir(parents=True)
    orphan_video.parent.mkdir(parents=True)
    orphan_annotation.write_bytes(b"stale")
    orphan_video.write_bytes(b"stale")

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
        clean_output=True,
    )

    assert not orphan_annotation.exists()
    assert not orphan_video.exists()


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
    assert "convert_undistorted)" in gigahands_script
    assert "stats_undistorted)" in gigahands_script
    assert "clean_generated)" in gigahands_script
    assert "train_undistorted)" in gigahands_script
    assert "eval_undistorted)" in gigahands_script
    assert "NCCL_P2P_DISABLE" in gigahands_script
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
            "statistics_dataset_name": "gigahands_real_train",
        }
    }

    kwargs = module.dataset_kwargs_from_config(config)

    assert kwargs["action_type"] == "keypoints"
    assert kwargs["use_rel"] is False
    assert kwargs["rel_mode"] == "step"
    assert kwargs["clip_len"] is None
    assert kwargs["state_mask_prob"] == 0.0
    assert kwargs["statistics_dataset_name"] == "gigahands_real_train"


def test_evaluator_prefers_episode_camera_metadata():
    module = load_module(EVAL_PATH, "evaluate_gigahands_stage1")
    epi = {
        "intrinsics": np.eye(3, dtype=np.float32),
        "camera": {
            "name": "cam0",
            "image_size": [1280, 720],
            "intrinsics": (np.eye(3, dtype=np.float32) * 2).tolist(),
            "original_intrinsics": np.eye(3, dtype=np.float32).tolist(),
            "distortion": [0.1, 0.01, 0.0, 0.0],
            "undistorted": True,
        },
    }
    context = {"episode": epi}

    info = module.camera_info_for_context(None, context)

    assert info["name"] == "cam0"
    assert info["image_size"] == (1280, 720)
    assert info["undistorted"] is True
    np.testing.assert_allclose(info["intrinsics"], np.eye(3, dtype=np.float32) * 2)


def test_rgb_overlay_gt_prefers_raw_mano_labels(monkeypatch):
    module = load_module(EVAL_PATH, "evaluate_gigahands_stage1")

    monkeypatch.setattr(
        module,
        "split_state_beta_122",
        lambda state: (
            (np.zeros(51, dtype=np.float32), np.zeros(10, dtype=np.float32)),
            (np.zeros(51, dtype=np.float32), np.zeros(10, dtype=np.float32)),
        ),
    )
    monkeypatch.setattr(module, "recon_traj_from_actions", lambda state, action: action)
    monkeypatch.setattr(
        module,
        "traj_to_mano_labels",
        lambda traj, beta: {
            "transl_worldspace": np.asarray(traj, dtype=np.float32),
            "global_orient_worldspace": np.zeros((traj.shape[0], 3), dtype=np.float32),
            "hand_pose": np.zeros((traj.shape[0], 45), dtype=np.float32),
            "beta": np.asarray(beta, dtype=np.float32),
        },
    )
    monkeypatch.setattr(module, "mano_vertices_from_labels", lambda mano, labels, is_left, device: labels)

    result_by_label = {
        "trained": {
            "states": [np.zeros(122, dtype=np.float32)],
            "unnormalized_targets": [np.ones((2, 102), dtype=np.float32) * 7.0],
            "unnormalized_predictions": [np.ones((2, 102), dtype=np.float32) * 5.0],
        }
    }
    context = {
        "local_frame_ids": np.array([0, 1], dtype=np.int64),
        "episode": {
            "left": {
                "transl_camspace": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "global_orient_camspace": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "hand_pose": np.zeros((2, 45), dtype=np.float32).tolist(),
                "beta": np.zeros(10, dtype=np.float32).tolist(),
            },
            "right": {
                "transl_camspace": [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                "global_orient_camspace": [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]],
                "hand_pose": np.zeros((2, 45), dtype=np.float32).tolist(),
                "beta": np.ones(10, dtype=np.float32).tolist(),
            },
        },
    }

    mesh_sets = module.build_mano_mesh_sets(
        result_by_label,
        "trained",
        0,
        mano=object(),
        device="cpu",
        context=context,
        gt_source="raw_mano",
    )

    np.testing.assert_allclose(mesh_sets["gt"]["left"]["transl_worldspace"], np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    np.testing.assert_allclose(mesh_sets["gt"]["right"]["transl_worldspace"], np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32))
    np.testing.assert_allclose(mesh_sets["trained"]["left"]["transl_worldspace"], np.ones((2, 51), dtype=np.float32) * 5.0)


def test_rgb_overlay_gt_falls_back_to_reconstructed_target_without_raw_mano(monkeypatch):
    module = load_module(EVAL_PATH, "evaluate_gigahands_stage1")

    monkeypatch.setattr(
        module,
        "split_state_beta_122",
        lambda state: (
            (np.zeros(51, dtype=np.float32), np.zeros(10, dtype=np.float32)),
            (np.zeros(51, dtype=np.float32), np.zeros(10, dtype=np.float32)),
        ),
    )
    monkeypatch.setattr(module, "recon_traj_from_actions", lambda state, action: action)
    monkeypatch.setattr(
        module,
        "traj_to_mano_labels",
        lambda traj, beta: {
            "transl_worldspace": np.asarray(traj, dtype=np.float32),
            "global_orient_worldspace": np.zeros((traj.shape[0], 3), dtype=np.float32),
            "hand_pose": np.zeros((traj.shape[0], 45), dtype=np.float32),
            "beta": np.asarray(beta, dtype=np.float32),
        },
    )
    monkeypatch.setattr(module, "mano_vertices_from_labels", lambda mano, labels, is_left, device: labels)

    target = np.ones((2, 102), dtype=np.float32) * 9.0
    result_by_label = {
        "trained": {
            "states": [np.zeros(122, dtype=np.float32)],
            "unnormalized_targets": [target],
            "unnormalized_predictions": [np.ones((2, 102), dtype=np.float32) * 4.0],
        }
    }
    context = {"local_frame_ids": np.array([0, 1], dtype=np.int64), "episode": {"left": {}, "right": {}}}

    mesh_sets = module.build_mano_mesh_sets(
        result_by_label,
        "trained",
        0,
        mano=object(),
        device="cpu",
        context=context,
        gt_source="raw_mano",
    )

    np.testing.assert_allclose(mesh_sets["gt"]["left"]["transl_worldspace"], np.ones((2, 51), dtype=np.float32) * 9.0)
