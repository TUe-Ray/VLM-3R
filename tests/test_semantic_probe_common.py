import importlib.util
import pathlib
import sys

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
sys.path.insert(0, str(PROBING_DIR))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


semantic = load_module("_semantic_probe_common", PROBING_DIR / "semantic_probe_common.py")
align = load_module("_check_semantic_feature_alignment", PROBING_DIR / "check_semantic_feature_alignment.py")
prepare = load_module("_prepare_semantic_probe_scannet", PROBING_DIR / "prepare_semantic_probe_scannet.py")


def write_mapping_files(tmp_path):
    tsv = tmp_path / "scannet-labels.combined.tsv"
    tsv.write_text(
        "id\traw_category\tnyu40id\n"
        "41\twall raw\t1\n"
        "42\tfloor raw\t2\n"
        "43\tchair raw\t5\n"
        "44\tunknown raw\t99\n",
        encoding="utf-8",
    )
    classes = tmp_path / "classes_SemVoxLabel-nyu40id.txt"
    classes.write_text(
        "\n".join(
            [
                "1 wall",
                "2 floor",
                "3 cabinet",
                "4 bed",
                "5 chair",
                "6 sofa",
                "7 table",
                "8 door",
                "9 window",
                "10 bookshelf",
                "11 picture",
                "12 counter",
                "14 desk",
                "16 curtain",
                "24 refrigerator",
                "28 shower curtain",
                "33 toilet",
                "34 sink",
                "36 bathtub",
                "39 otherfurniture",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return tsv, classes


def test_label_space_inference_distinguishes_raw_and_nyu40():
    raw_to_nyu40 = {41: 1, 42: 2, 43: 5}
    assert semantic.infer_label_value_space([41, 42], raw_to_nyu40=raw_to_nyu40) == "raw_id"
    assert semantic.infer_label_value_space([1, 2, 5], raw_to_nyu40=raw_to_nyu40) == "nyu40id"


def test_label_space_inference_stops_on_ambiguous_values():
    raw_to_nyu40 = {1: 5, 41: 1}
    try:
        semantic.infer_label_value_space([1], raw_to_nyu40=raw_to_nyu40)
    except ValueError as exc:
        assert "Ambiguous" in str(exc)
    else:
        raise AssertionError("Expected ambiguous label space to raise")


def test_source_value_counts_logs_actual_values():
    videos = [{"source_dataset": "scannet"}, {"source_dataset": "arkitscenes"}, {"source_dataset": "scannet"}]
    assert prepare.source_value_counts(videos, "source_dataset") == {"arkitscenes": 1, "scannet": 2}


def test_mapping_sends_non_scannet20_to_ignore(tmp_path):
    tsv, classes = write_mapping_files(tmp_path)
    mapping = semantic.build_label_mapping(label_tsv=tsv, scannet20_class_file=classes, label_value_space="raw_id")
    labels = torch.tensor([[41, 42], [43, 44]])
    mapped = semantic.map_label_tensor_to_train_labels(labels, mapping)
    assert mapped.tolist() == [[0, 1], [4, semantic.IGNORE_INDEX]]


def test_class_file_supports_id_only_lines_with_tsv_name_fallback(tmp_path):
    tsv, _classes = write_mapping_files(tmp_path)
    id_only = tmp_path / "id_only_classes.txt"
    id_only.write_text(
        "\n".join(str(x) for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]) + "\n",
        encoding="utf-8",
    )
    mapping = semantic.build_label_mapping(label_tsv=tsv, scannet20_class_file=id_only, label_value_space="raw_id")
    assert mapping["class_index_to_name"][0] == "wall raw"
    assert mapping["class_index_to_name"][2] == "class_3"


def test_majority_vote_uses_only_valid_train_labels():
    labels = torch.tensor(
        [
            [0, 0, semantic.IGNORE_INDEX, semantic.IGNORE_INDEX],
            [1, 0, semantic.IGNORE_INDEX, semantic.IGNORE_INDEX],
            [2, 2, 3, 3],
            [semantic.IGNORE_INDEX, 2, 3, 4],
        ]
    )
    pooled = semantic.downsample_train_labels_majority(labels, (2, 2), num_classes=20)
    assert pooled.tolist() == [[0, semantic.IGNORE_INDEX], [2, 3]]


def test_miou_excludes_absent_classes():
    confusion = torch.zeros((20, 20), dtype=torch.long)
    confusion[0, 0] = 3
    confusion[1, 1] = 1
    metrics = semantic.metrics_from_confusion(confusion)
    assert metrics["present_classes"] == [0, 1]
    assert metrics["num_present_classes"] == 2
    assert metrics["num_gt_present_classes"] == 2
    assert metrics["mIoU"] == 1.0
    assert metrics["mIoU_gt_present"] == 1.0
    assert metrics["per_class_IoU"][2] is None


def test_alignment_rejects_mismatched_feature_grid():
    ok, layout = align.check_feature_shape(torch.randn(14, 14, 8), (14, 14))
    assert ok and layout == "grid_tensor"
    ok, layout = align.check_feature_shape(torch.randn(1, 14, 14, 8), (14, 14))
    assert ok and layout == "grid_tensor"
    ok, layout = align.check_feature_shape(torch.randn(1, 196, 8), (14, 14))
    assert ok and layout == "flat_tokens"
    ok, reason = align.check_feature_shape(torch.randn(195, 8), (14, 14))
    assert not ok
    assert "token_count_mismatch" in reason
