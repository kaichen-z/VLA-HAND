import importlib.util
import sys
from pathlib import Path


import numpy as np

from test_gigahands_stage1_converter import load_converter, write_demo_fixture


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = REPO_ROOT / "tools" / "analyze_hand_usage.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_hand_usage", ANALYZER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def convert_demo_fixture(tmp_path: Path):
    converter = load_converter()
    input_root = tmp_path / "gigahands_demo_all"
    output_root = tmp_path / "vitra_gigahands_demo"
    write_demo_fixture(input_root, frame_count=20)
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
    return input_root, output_root


def test_vitra_format_analysis_counts_converted_demo_as_dual_hand(tmp_path):
    analyzer = load_analyzer()
    _, output_root = convert_demo_fixture(tmp_path)

    result = analyzer.analyze_vitra_format_dataset(
        dataset_root=output_root,
        dataset_names=["gigahands"],
        sample_frames_per_dataset=-1,
        future_window=3,
    )

    summary = result["datasets"]["gigahands"]
    assert summary["episodes"] == 1
    assert summary["frames_sampled"] == 20
    assert summary["schema_dual_hand_frames"] == 20
    assert summary["valid_dual_hand_frames"] == 20
    assert summary["semantic_dual_hand_frames"] == 20
    assert summary["active_dual_hand_frames_by_threshold_cm"]["1.0"] > 0
    assert len(result["samples"]) == 20


def test_raw_gigahands_demo_analysis_counts_params_for_both_hands(tmp_path):
    analyzer = load_analyzer()
    input_root, _ = convert_demo_fixture(tmp_path)

    result = analyzer.analyze_gigahands_raw_demo(input_root)

    summary = result["datasets"]["gigahands_raw_demo"]
    assert summary["episodes"] == 1
    assert summary["frames_sampled"] == 20
    assert summary["schema_dual_hand_frames"] == 20
    assert summary["valid_dual_hand_frames"] == 20
    assert summary["active_dual_hand_frames_by_threshold_cm"]["1.0"] == 20
    assert len(result["samples"]) == 20


def test_write_outputs_creates_json_csv_and_markdown(tmp_path):
    analyzer = load_analyzer()
    _, output_root = convert_demo_fixture(tmp_path)
    result = analyzer.analyze_vitra_format_dataset(
        dataset_root=output_root,
        dataset_names=["gigahands"],
        sample_frames_per_dataset=-1,
        future_window=3,
    )
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "samples.csv"
    output_md = tmp_path / "report.md"

    analyzer.write_outputs(result, output_json=output_json, output_csv=output_csv, output_md=output_md)

    assert output_json.exists()
    assert output_csv.exists()
    assert output_md.exists()
    assert "Dataset" in output_md.read_text()
    assert "gigahands" in output_csv.read_text()
