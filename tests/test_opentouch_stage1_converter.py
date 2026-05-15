import importlib.util
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = REPO_ROOT / "data" / "preprocessing" / "convert_opentouch_to_vitra_stage1.py"


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_opentouch_to_vitra_stage1", CONVERTER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def encode_jpeg(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    assert ok
    return encoded.tobytes()


def write_opentouch_fixture(
    root: Path,
    frame_count: int = 20,
    include_left: bool = False,
    clip_id: str = "clip_000001",
    separate_touch_timestamps: bool = False,
    h5_stem: str = "synthetic_session",
    timestamps: np.ndarray | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    h5_path = root / f"{h5_stem}.hdf5"
    frames = []
    for idx in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[..., 0] = idx
        frame[..., 1] = 30
        frame[..., 2] = 200
        frames.append(np.frombuffer(encode_jpeg(frame), dtype=np.uint8))

    landmarks = np.zeros((frame_count, 21, 3), dtype=np.float32)
    landmarks[:, :, 2] = 1.0
    landmarks[:, :, 0] = np.linspace(0.0, 0.1, frame_count)[:, None]
    landmarks[:, 0, :] += np.array([0.2, 0.0, 0.0], dtype=np.float32)
    left_landmarks = landmarks.copy()
    left_landmarks[:, :, 0] *= -1.0

    with h5py.File(h5_path, "w") as handle:
        clip = handle.create_group("data").create_group(clip_id)
        vlen = h5py.vlen_dtype(np.dtype("uint8"))
        jpeg_ds = clip.create_dataset("rgb_images_jpeg", (frame_count,), dtype=vlen)
        for idx, encoded in enumerate(frames):
            jpeg_ds[idx] = encoded
        clip.create_dataset("right_hand_landmarks", data=landmarks)
        if separate_touch_timestamps:
            right_pressure = np.arange(frame_count, dtype=np.float32)[:, None, None] * np.ones((frame_count, 16, 16), dtype=np.float32)
        else:
            right_pressure = np.ones((frame_count, 16, 16), dtype=np.float32)
        clip.create_dataset("right_pressure", data=right_pressure)
        if include_left:
            clip.create_dataset("left_hand_landmarks", data=left_landmarks)
            clip.create_dataset("left_pressure", data=np.full((frame_count, 16, 16), 2.0, dtype=np.float32))
        if timestamps is None:
            timestamps = np.arange(frame_count, dtype=np.float64) / 30.0
        clip.create_dataset("timestamps", data=np.asarray(timestamps))
        if separate_touch_timestamps:
            touch_timestamps = np.arange(frame_count, dtype=np.float64) / 30.0
            touch_timestamps[5:] += 0.004
            touch_timestamps[-1] += 1.0
            clip.create_dataset("touch_timestamps", data=touch_timestamps)

    return h5_path


def test_opentouch_conversion_matches_labels_by_timestamp_when_demo_ids_repeat(tmp_path):
    converter = load_converter()
    input_root = tmp_path / "opentouch_raw"
    output_root = tmp_path / "vitra_opentouch_keypoint"
    write_opentouch_fixture(
        input_root,
        clip_id="demo_000",
        h5_stem="session_a",
        timestamps=np.arange(1000, 1020, dtype=np.int64),
    )
    write_opentouch_fixture(
        input_root,
        clip_id="demo_000",
        h5_stem="session_b",
        timestamps=np.arange(2000, 2020, dtype=np.int64),
    )
    labels = input_root / "labels.csv"
    labels.write_text(
        "\n".join(
            [
                "clip_id,object_name,object_category,environment,action,grip_type,description,ts_start,ts_end",
                "merged::demo_000,block,tool,lab,pressing,Tip Pinch,Timestamp description for session A.,1000,1019",
                "merged::demo_001,cup,cup,kitchen,picking up,Small Diameter,Timestamp description for session B.,2000,2019",
            ]
        ),
        encoding="utf-8",
    )

    report = converter.convert_opentouch_to_vitra_stage1(
        opentouch_root=input_root,
        output_root=output_root,
        labels_path=labels,
        min_frames=17,
        train_ratio=1.0,
        write_video=False,
        require_labels=True,
    )

    assert report["num_episodes_written"] == 2
    episodes = {}
    for episode_path in (output_root / "Annotation" / "opentouch_keypoint_train" / "episodic_annotations").glob("*.npy"):
        episode = np.load(episode_path, allow_pickle=True).item()
        h5_stem = Path(episode["opentouch"]["h5_path"]).stem
        episodes[h5_stem] = episode

    assert episodes["session_a"]["text"]["right"][0][0] == "Timestamp description for session A."
    assert episodes["session_a"]["opentouch"]["label_clip_id"] == "merged::demo_000"
    assert episodes["session_a"]["opentouch"]["object_name"] == "block"
    assert episodes["session_a"]["opentouch"]["label_match_method"] == "timestamp"
    assert episodes["session_b"]["text"]["right"][0][0] == "Timestamp description for session B."
    assert episodes["session_b"]["opentouch"]["label_clip_id"] == "merged::demo_001"
    assert episodes["session_b"]["opentouch"]["object_name"] == "cup"
    assert episodes["session_b"]["opentouch"]["label_match_method"] == "timestamp"


def test_opentouch_conversion_require_labels_rejects_unmatched_clip(tmp_path):
    converter = load_converter()
    input_root = tmp_path / "opentouch_raw"
    output_root = tmp_path / "vitra_opentouch_keypoint"
    write_opentouch_fixture(
        input_root,
        clip_id="demo_000",
        h5_stem="session_a",
        timestamps=np.arange(1000, 1020, dtype=np.int64),
    )
    labels = input_root / "labels.csv"
    labels.write_text(
        "\n".join(
            [
                "clip_id,description,ts_start,ts_end",
                "merged::demo_000,Wrong timestamp description.,3000,3019",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No label payload"):
        converter.convert_opentouch_to_vitra_stage1(
            opentouch_root=input_root,
            output_root=output_root,
            labels_path=labels,
            min_frames=17,
            train_ratio=1.0,
            write_video=False,
            require_labels=True,
        )


def test_opentouch_conversion_writes_keypoint_stage1_episode(tmp_path):
    converter = load_converter()
    input_root = tmp_path / "opentouch_raw"
    output_root = tmp_path / "vitra_opentouch_keypoint"
    write_opentouch_fixture(input_root)
    labels = input_root / "labels.json"
    labels.write_text(json.dumps({"clip_000001": "Press the textured surface with the right hand."}))

    report = converter.convert_opentouch_to_vitra_stage1(
        opentouch_root=input_root,
        output_root=output_root,
        labels_path=labels,
        min_frames=17,
        train_ratio=1.0,
        write_video=True,
    )

    assert report["num_episodes_written"] == 1
    assert report["num_frames_written"] == 20
    assert (output_root / "Annotation" / "opentouch_keypoint_train" / "episode_frame_index.npz").exists()
    assert (output_root / "Video" / "OpenTouch_root" / "synthetic_session" / "clip_000001.mp4").exists()

    episode_path = next((output_root / "Annotation" / "opentouch_keypoint_train" / "episodic_annotations").glob("*.npy"))
    epi = np.load(episode_path, allow_pickle=True).item()
    assert epi["video_name"] == "synthetic_session/clip_000001.mp4"
    assert epi["anno_type"] == "right"
    assert epi["right"]["joints_worldspace"].shape == (20, 21, 3)
    assert epi["right"]["kept_frames"].all()
    assert not epi["left"]["kept_frames"].any()
    assert epi["text"]["right"][0][0] == "Press the textured surface with the right hand."
    assert epi["opentouch"]["right_pressure"].shape == (20, 16, 16)
    assert epi["opentouch"]["touch_pressure"].shape == (20, 2, 16, 16)
    assert epi["opentouch"]["touch_mask"].shape == (20, 2)
    assert not epi["opentouch"]["touch_mask"][:, 0].any()
    assert epi["opentouch"]["touch_mask"][:, 1].all()
    assert "video_timestamps" in epi["opentouch"]
    assert "touch_timestamps" in epi["opentouch"]
    assert "touch_alignment_valid" in epi["opentouch"]


def test_opentouch_conversion_preserves_both_hand_touch_and_manifest(tmp_path):
    converter = load_converter()
    input_root = tmp_path / "opentouch_raw"
    output_root = tmp_path / "vitra_opentouch_keypoint"
    manifest_path = tmp_path / "contact_manifest.jsonl"
    write_opentouch_fixture(input_root, include_left=True, clip_id="clip_keep")
    labels = input_root / "labels.csv"
    labels.write_text("clip_id,action,object_name\nclip_keep,grip,block\n", encoding="utf-8")

    report = converter.convert_opentouch_to_vitra_stage1(
        opentouch_root=input_root,
        output_root=output_root,
        labels_path=labels,
        min_frames=17,
        train_ratio=1.0,
        write_video=True,
        filter_contact_keywords=True,
        contact_manifest_path=manifest_path,
    )

    assert report["num_clips_seen"] == 1
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert records[0]["clip_id"] == "clip_keep"
    assert records[0]["matched_keyword"] == "grip"

    episode_path = next((output_root / "Annotation" / "opentouch_keypoint_train" / "episodic_annotations").glob("*.npy"))
    epi = np.load(episode_path, allow_pickle=True).item()
    assert epi["left"]["kept_frames"].all()
    assert epi["right"]["kept_frames"].all()
    assert epi["opentouch"]["left_pressure"].shape == (20, 16, 16)
    assert epi["opentouch"]["right_pressure"].shape == (20, 16, 16)
    assert np.allclose(epi["opentouch"]["touch_pressure"][:, 0], 2.0)
    assert np.allclose(epi["opentouch"]["touch_pressure"][:, 1], 1.0)
    assert epi["opentouch"]["touch_mask"].all()


def test_opentouch_contact_filter_skips_nonmatching_labels(tmp_path):
    converter = load_converter()
    input_root = tmp_path / "opentouch_raw"
    output_root = tmp_path / "vitra_opentouch_keypoint"
    manifest_path = tmp_path / "contact_manifest.jsonl"
    write_opentouch_fixture(input_root, clip_id="clip_skip")
    labels = input_root / "labels.json"
    labels.write_text(json.dumps({"clip_skip": {"action": "observe", "object_name": "surface"}}))

    report = converter.convert_opentouch_to_vitra_stage1(
        opentouch_root=input_root,
        output_root=output_root,
        labels_path=labels,
        min_frames=17,
        train_ratio=1.0,
        write_video=False,
        filter_contact_keywords=True,
        contact_manifest_path=manifest_path,
    )

    assert report["num_clips_seen"] == 0
    assert report["num_episodes_written"] == 0
    assert manifest_path.read_text() == ""


def test_converted_opentouch_can_be_read_as_keypoint_frame_dataset(tmp_path):
    converter = load_converter()
    input_root = tmp_path / "opentouch_raw"
    output_root = tmp_path / "vitra_opentouch_keypoint"
    write_opentouch_fixture(input_root)

    converter.convert_opentouch_to_vitra_stage1(
        opentouch_root=input_root,
        output_root=output_root,
        min_frames=17,
        train_ratio=1.0,
        write_video=True,
    )

    from vitra.datasets.dataset import FrameDataset

    ds = FrameDataset(
        dataset_folder=str(output_root),
        dataset_name="opentouch_keypoint_train",
        action_future_window_size=16,
        action_type="keypoints",
        augmentation=False,
        load_images=False,
        state_mask_prob=0.0,
    )
    item = ds[0]

    assert len(ds) == 20
    assert item["action_list"].shape == (17, 138)
    assert item["current_state"].shape == (158,)
    assert not item["action_mask"][:, 0].any()
    assert item["action_mask"][:, 1].all()
    assert item["current_state_mask"].tolist() == [False, True]


class FakeProcessorOutput(dict):
    def to(self, dtype):
        return FakeProcessorOutput({key: value.to(dtype) if torch.is_floating_point(value) else value for key, value in self.items()})


class FakeProcessor:
    def __call__(self, text, images, return_tensors):
        assert text.startswith("<image>")
        assert len(images) == 1
        return FakeProcessorOutput(
            {
                "input_ids": torch.ones((1, 8), dtype=torch.long),
                "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
            }
        )


def test_converted_opentouch_can_be_read_with_images_and_padded_for_training(tmp_path):
    converter = load_converter()
    input_root = tmp_path / "opentouch_raw"
    output_root = tmp_path / "vitra_opentouch_keypoint"
    write_opentouch_fixture(input_root)

    converter.convert_opentouch_to_vitra_stage1(
        opentouch_root=input_root,
        output_root=output_root,
        min_frames=17,
        train_ratio=1.0,
        write_video=True,
    )

    from vitra.datasets.dataset import FrameDataset

    ds = FrameDataset(
        dataset_folder=str(output_root),
        dataset_name="opentouch_keypoint_train",
        action_future_window_size=16,
        action_type="keypoints",
        augmentation=False,
        normalization=False,
        processor=FakeProcessor(),
        load_images=True,
        state_mask_prob=0.0,
    )
    item = ds[0]

    assert item["actions"].shape == (17, 192)
    assert item["current_state"].shape == (212,)
    assert item["action_masks"].shape == (17, 192)
    assert item["current_state_mask"].shape == (212,)
    assert item["pixel_values"].shape == (1, 3, 224, 224)


def test_opentouch_dataset_registration_mentions_keypoint_paths():
    dataset_py = (REPO_ROOT / "vitra" / "datasets" / "dataset.py").read_text()
    mixtures_py = (REPO_ROOT / "vitra" / "datasets" / "data_mixture.py").read_text()
    stats_py = (REPO_ROOT / "vitra" / "datasets" / "calculate_statistics.py").read_text()

    assert "dataset_name.startswith('opentouch_keypoint_')" in dataset_py
    assert "Video/OpenTouch_root" in dataset_py
    assert '"opentouch_keypoint_train"' in mixtures_py
    assert '"opentouch_keypoint_test"' in mixtures_py
    assert "opentouch_keypoint_train" in stats_py


def test_opentouch_conversion_aligns_touch_timestamps_with_tolerance(tmp_path):
    converter = load_converter()
    input_root = tmp_path / "opentouch_raw"
    output_root = tmp_path / "vitra_opentouch_keypoint"
    write_opentouch_fixture(input_root, separate_touch_timestamps=True)

    report = converter.convert_opentouch_to_vitra_stage1(
        opentouch_root=input_root,
        output_root=output_root,
        min_frames=17,
        train_ratio=1.0,
        write_video=False,
        touch_alignment_tolerance=0.01,
    )

    assert report["num_episodes_written"] == 1
    episode_path = next((output_root / "Annotation" / "opentouch_keypoint_train" / "episodic_annotations").glob("*.npy"))
    epi = np.load(episode_path, allow_pickle=True).item()
    touch = epi["opentouch"]
    assert touch["touch_alignment_valid"][:-1].all()
    assert not touch["touch_alignment_valid"][-1]
    assert touch["touch_aligned_indices"][:5].tolist() == [0, 1, 2, 3, 4]
    assert np.allclose(touch["touch_pressure"][:5, 1, 0, 0], [0, 1, 2, 3, 4])
    assert not touch["touch_mask"][-1].any()


def test_opentouch_conversion_normalizes_nanosecond_timestamps(tmp_path):
    converter = load_converter()
    raw = np.array([1_000_000_000, 1_033_333_333, 1_066_666_666], dtype=np.int64)
    seconds = converter.normalize_timestamps_seconds(raw)
    assert np.allclose(np.diff(seconds), [1.0 / 30.0, 1.0 / 30.0], atol=1e-3)


def test_opentouch_keypoint_statistics_handle_masked_left_hand(tmp_path):
    converter = load_converter()
    input_root = tmp_path / "opentouch_raw"
    output_root = tmp_path / "vitra_opentouch_keypoint"
    stats_root = output_root / "Annotation" / "statistics"
    write_opentouch_fixture(input_root)

    converter.convert_opentouch_to_vitra_stage1(
        opentouch_root=input_root,
        output_root=output_root,
        min_frames=17,
        train_ratio=1.0,
        write_video=True,
    )

    from vitra.datasets.calculate_statistics import compute_statistics
    from vitra.datasets.dataset import FrameDataset

    ds = FrameDataset(
        dataset_folder=str(output_root),
        dataset_name="opentouch_keypoint_train",
        action_future_window_size=0,
        action_type="keypoints",
        augmentation=False,
        load_images=False,
        state_mask_prob=0.0,
    )
    compute_statistics(ds, num_workers=0, batch_size=8, save_folder=str(stats_root))

    stats = json.loads((stats_root / "opentouch_keypoint_train_keypoints_statistics.json").read_text())
    assert len(stats["action_right"]["mean"]) == 69
    assert len(stats["state_right"]["mean"]) == 79
    assert not np.isnan(np.asarray(stats["action_left"]["mean"], dtype=np.float32)).any()
    assert np.allclose(stats["action_left"]["mean"], 0.0)
    assert np.allclose(stats["action_left"]["std"], 1.0)
