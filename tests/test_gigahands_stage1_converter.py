import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = REPO_ROOT / "data" / "preprocessing" / "convert_gigahands_to_vitra_stage1.py"


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_gigahands_to_vitra_stage1", CONVERTER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_demo_fixture(root: Path, frame_count: int = 20) -> Path:
    sequence_root = root / "hand_pose" / "p003-instrument_0033"
    params_root = sequence_root / "params"
    keypoints_root = sequence_root / "keypoints_3d_mano"
    video_root = sequence_root / "rgb_vid" / "brics-odroid-011_cam0"
    params_root.mkdir(parents=True)
    keypoints_root.mkdir(parents=True)
    video_root.mkdir(parents=True)

    (video_root / "brics-odroid-011_cam0_000033.mp4").write_bytes(b"fake mp4")
    (video_root / "brics-odroid-011_cam0_000033.txt").write_text("Pick up the small instrument.\n")

    line = [
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
        "brics-odroid-011_cam0",
        1.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.2,
        0.3,
    ]
    (sequence_root / "optim_params.txt").write_text(" ".join(map(str, line)) + "\n")

    poses = np.zeros((frame_count, 48), dtype=np.float32)
    poses[:, 3] = np.linspace(0.0, 0.1, frame_count)
    th_left = np.stack(
        [
            np.linspace(0.0, 0.2, frame_count),
            np.zeros(frame_count),
            np.ones(frame_count),
        ],
        axis=1,
    ).astype(np.float32)
    th_right = th_left + np.array([0.1, 0.0, 0.0], dtype=np.float32)
    params = {
        "left": {
            "poses": poses.tolist(),
            "Rh": np.zeros((frame_count, 3), dtype=np.float32).tolist(),
            "Th": th_left.tolist(),
            "shapes": np.zeros((frame_count, 10), dtype=np.float32).tolist(),
        },
        "right": {
            "poses": poses.tolist(),
            "Rh": np.zeros((frame_count, 3), dtype=np.float32).tolist(),
            "Th": th_right.tolist(),
            "shapes": np.zeros((frame_count, 10), dtype=np.float32).tolist(),
        },
    }
    (params_root / "033.json").write_text(json.dumps(params))

    left_joints = np.zeros((frame_count, 21, 3), dtype=np.float32)
    right_joints = np.zeros((frame_count, 21, 3), dtype=np.float32)
    left_joints[:, :, 2] = 1.0
    right_joints[:, :, 0] = 0.1
    right_joints[:, :, 2] = 1.0
    keypoints = {"left": left_joints.tolist(), "right": right_joints.tolist()}
    (keypoints_root / "033.json").write_text(json.dumps(keypoints))

    return sequence_root


def test_demo_layout_conversion_writes_vitra_episode(tmp_path):
    converter = load_converter()
    input_root = tmp_path / "gigahands_demo_all"
    output_root = tmp_path / "vitra_gigahands_demo"
    write_demo_fixture(input_root)

    report = converter.convert_gigahands_to_vitra(
        gigahands_root=input_root,
        output_root=output_root,
        input_layout="demo",
        camera="brics-odroid-011_cam0",
        min_frames=17,
        min_valid_ratio=0.9,
        write_video=True,
        undistort=False,
    )

    assert report["num_episodes_written"] == 1
    assert (output_root / "Annotation" / "gigahands" / "episode_frame_index.npz").exists()
    episodes = list((output_root / "Annotation" / "gigahands" / "episodic_annotations").glob("*.npy"))
    assert len(episodes) == 1

    epi = np.load(episodes[0], allow_pickle=True).item()
    assert epi["video_name"].endswith(".mp4")
    assert epi["video_decode_frame"].shape == (20,)
    assert epi["intrinsics"].shape == (3, 3)
    assert epi["extrinsics"].shape == (20, 4, 4)
    assert epi["left"]["hand_pose"].shape == (20, 15, 3, 3)
    assert epi["right"]["joints_worldspace"].shape == (20, 21, 3)
    assert epi["left"]["kept_frames"].dtype == np.bool_
    assert epi["text"]["left"][0][0] == "Pick up the small instrument."


def test_short_demo_sequence_is_skipped_and_reported(tmp_path):
    converter = load_converter()
    input_root = tmp_path / "gigahands_demo_all"
    output_root = tmp_path / "vitra_gigahands_demo"
    write_demo_fixture(input_root, frame_count=8)

    report = converter.convert_gigahands_to_vitra(
        gigahands_root=input_root,
        output_root=output_root,
        input_layout="demo",
        camera="brics-odroid-011_cam0",
        min_frames=17,
        min_valid_ratio=0.9,
        write_video=True,
        undistort=False,
    )

    assert report["num_episodes_written"] == 0
    assert report["skipped_short_clip"] == 1


def test_demo_sequence_without_params_is_reported(tmp_path):
    converter = load_converter()
    input_root = tmp_path / "gigahands_demo_all"
    output_root = tmp_path / "vitra_gigahands_demo"
    sequence_root = input_root / "hand_pose" / "p041-plant-0004"
    video_root = sequence_root / "rgb_vid" / "brics-odroid-011_cam0"
    video_root.mkdir(parents=True)
    (sequence_root / "params").mkdir()
    (video_root / "brics-odroid-011_cam0_000004.mp4").write_bytes(b"fake mp4")
    (sequence_root / "optim_params.txt").write_text(
        "11 1280 720 900 900 640 360 0 0 0 0 brics-odroid-011_cam0 1 0 0 0 0 0 0\n"
    )

    report = converter.convert_gigahands_to_vitra(
        gigahands_root=input_root,
        output_root=output_root,
        input_layout="demo",
        camera="brics-odroid-011_cam0",
        min_frames=17,
        min_valid_ratio=0.9,
        write_video=True,
        undistort=False,
    )

    assert report["num_sequences_seen"] == 1
    assert report["num_annotations_seen"] == 1
    assert report["num_episodes_written"] == 0
    assert report["skipped_missing_params"] == 1


def test_demo_frame_timestamp_txt_is_not_used_as_instruction(tmp_path):
    converter = load_converter()
    input_root = tmp_path / "gigahands_demo_all"
    output_root = tmp_path / "vitra_gigahands_demo"
    sequence_root = write_demo_fixture(input_root)
    txt_path = next((sequence_root / "rgb_vid" / "brics-odroid-011_cam0").glob("*.txt"))
    txt_path.write_text("frame_1726962101765814_000000000000\nframe_1726962101801743_000000000001\n")

    converter.convert_gigahands_to_vitra(
        gigahands_root=input_root,
        output_root=output_root,
        input_layout="demo",
        camera="brics-odroid-011_cam0",
        min_frames=17,
        min_valid_ratio=0.9,
        write_video=True,
        undistort=False,
    )

    episode = next((output_root / "Annotation" / "gigahands" / "episodic_annotations").glob("*.npy"))
    epi = np.load(episode, allow_pickle=True).item()
    assert epi["text"]["left"][0][0] == "p003 instrument 0033."
    assert not epi["text"]["left"][0][0].startswith("frame_")


def test_vitra_dataset_registration_mentions_gigahands_paths():
    dataset_py = (REPO_ROOT / "vitra" / "datasets" / "dataset.py").read_text()
    mixtures_py = (REPO_ROOT / "vitra" / "datasets" / "data_mixture.py").read_text()
    human_dataset_py = (REPO_ROOT / "vitra" / "datasets" / "human_dataset.py").read_text()

    assert "dataset_name == 'gigahands'" in dataset_py
    assert "Annotation/gigahands/episode_frame_index.npz" in dataset_py
    assert "Video/GigaHands_root" in dataset_py
    assert '"gigahands_demo_only"' in mixtures_py
    assert "dataset_name == 'GigaHands'" in human_dataset_py


def test_converted_demo_can_be_read_by_frame_dataset_without_images(tmp_path):
    converter = load_converter()
    input_root = tmp_path / "gigahands_demo_all"
    output_root = tmp_path / "vitra_gigahands_demo"
    write_demo_fixture(input_root)

    converter.convert_gigahands_to_vitra(
        gigahands_root=input_root,
        output_root=output_root,
        input_layout="demo",
        camera="brics-odroid-011_cam0",
        min_frames=17,
        min_valid_ratio=0.9,
        write_video=True,
        undistort=False,
    )

    from vitra.datasets.dataset import FrameDataset

    ds = FrameDataset(
        dataset_folder=str(output_root),
        dataset_name="gigahands",
        action_future_window_size=16,
        augmentation=False,
        load_images=False,
        state_mask_prob=0.0,
    )
    item = ds[0]

    assert len(ds) == 20
    assert item["action_list"].shape == (17, 102)
    assert item["current_state"].shape == (122,)
    assert item["action_mask"].shape == (17, 2)
    assert item["current_state_mask"].shape == (2,)
    assert item["intrinsics"].shape == (3, 3)
