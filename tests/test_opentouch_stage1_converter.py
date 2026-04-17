import importlib.util
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
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


def write_opentouch_fixture(root: Path, frame_count: int = 20) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    h5_path = root / "synthetic_session.hdf5"
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

    with h5py.File(h5_path, "w") as handle:
        clip = handle.create_group("data").create_group("clip_000001")
        vlen = h5py.vlen_dtype(np.dtype("uint8"))
        jpeg_ds = clip.create_dataset("rgb_images_jpeg", (frame_count,), dtype=vlen)
        for idx, encoded in enumerate(frames):
            jpeg_ds[idx] = encoded
        clip.create_dataset("right_hand_landmarks", data=landmarks)
        clip.create_dataset("right_pressure", data=np.ones((frame_count, 16, 16), dtype=np.float32))
        clip.create_dataset("timestamps", data=np.arange(frame_count, dtype=np.float64) / 30.0)

    return h5_path


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
