"""Dataset validation, leakage-aware splitting, and immutable YOLO export."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import yaml
from PIL import Image, UnidentifiedImageError

from frameforge.media import _RecentHashes, _dhash
from frameforge.project import load_project


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _config(
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    segment_seconds: float,
    negative_ratio: float,
) -> Dict[str, Any]:
    if split_mode not in {"source", "segment"}:
        raise ValueError("Cách chia dataset phải là theo video nguồn hoặc theo đoạn video.")
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    for name, value in ratios.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Tỷ lệ train/val/test không hợp lệ.")
        if value < 0 or value > 1:
            raise ValueError("Mỗi tỷ lệ train/val/test phải từ 0 đến 1.")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-6) or train_ratio <= 0:
        raise ValueError("Tổng tỷ lệ train/val/test phải bằng 100% và train phải lớn hơn 0%.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed phải là số nguyên từ 0 đến 4294967295.")
    if isinstance(segment_seconds, bool) or not isinstance(segment_seconds, (int, float)) or not math.isfinite(segment_seconds):
        raise ValueError("Độ dài đoạn video không hợp lệ.")
    if not 1 <= segment_seconds <= 3600:
        raise ValueError("Độ dài đoạn video phải từ 1 đến 3600 giây.")
    if isinstance(negative_ratio, bool) or not isinstance(negative_ratio, (int, float)) or not math.isfinite(negative_ratio):
        raise ValueError("Tỷ lệ ảnh negative phải là số từ 0 đến 100%.")
    if not 0 <= negative_ratio <= 1:
        raise ValueError("Tỷ lệ ảnh negative phải từ 0 đến 100%.")
    return {
        "split_mode": split_mode,
        "ratios": ratios,
        "seed": seed,
        "segment_seconds": float(segment_seconds),
        "negative_ratio": float(negative_ratio),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_parent_asset(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Parent dataset thiếu đường dẫn {0} tương đối hợp lệ.".format(label))
    relative = Path(value)
    if ".." in relative.parts:
        raise ValueError("Parent dataset có đường dẫn {0} vượt khỏi thư mục phiên bản.".format(label))
    try:
        asset = (root / relative).resolve(strict=True)
        asset.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError("Không đọc được {0} trong parent dataset: {1}".format(label, value)) from error
    if not asset.is_file():
        raise ValueError("Parent dataset thiếu tệp {0}: {1}".format(label, value))
    return asset


def _read_parent_dataset(manifest_value: str, class_names: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    manifest_path = Path(manifest_value).expanduser().resolve(strict=True)
    if not manifest_path.is_file() or manifest_path.name.casefold() != "manifest.json":
        raise ValueError("Chọn manifest.json của một phiên bản FrameForge đã xuất.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Không đọc được manifest parent: {0}".format(error)) from error
    if not isinstance(manifest, dict) or manifest.get("format") != "FrameForge YOLO dataset":
        raise ValueError("Parent phải là dataset YOLO do FrameForge tạo.")
    dataset_id = manifest.get("dataset_id")
    version = manifest.get("version")
    if not isinstance(dataset_id, str) or not dataset_id or isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Manifest parent thiếu dataset ID hoặc version hợp lệ.")
    expected_mapping = {str(index): name for index, name in enumerate(class_names)}
    if manifest.get("class_mapping") != expected_mapping:
        raise ValueError("Class schema của parent không khớp dự án hiện tại.")
    parent_split = manifest.get("split")
    if not isinstance(parent_split, dict) or parent_split.get("mode") != config["split_mode"]:
        raise ValueError("Cách chia của parent phải khớp cách chia hiện tại.")
    if config["split_mode"] == "segment" and not math.isclose(
        float(parent_split.get("segment_seconds", -1)), config["segment_seconds"], abs_tol=1e-9
    ):
        raise ValueError("Độ dài đoạn của parent phải khớp cấu hình hiện tại.")
    frozen_test = manifest.get("frozen_test_set")
    fixed_validation = manifest.get("fixed_validation_set")
    if not isinstance(frozen_test, dict) or not frozen_test.get("enabled"):
        raise ValueError("Parent chưa có test set cố định đã review; hãy chọn phiên bản đã đóng băng val/test.")
    if not isinstance(fixed_validation, dict) or not fixed_validation.get("enabled"):
        raise ValueError("Parent chưa có validation set cố định.")

    parent_root = manifest_path.parent.resolve(strict=True)
    rows = manifest.get("images")
    if not isinstance(rows, list):
        raise ValueError("Manifest parent thiếu danh sách ảnh.")
    by_split: Dict[str, List[Dict[str, Any]]] = {split: [] for split in SPLITS}
    digest_splits: Dict[str, str] = {}
    group_splits: Dict[str, str] = {}
    valid_class_ids = set(range(len(class_names)))
    for row in rows:
        if not isinstance(row, dict) or row.get("split") not in SPLITS:
            raise ValueError("Manifest parent có bản ghi ảnh hoặc split không hợp lệ.")
        split = row["split"]
        digest = row.get("sha256")
        group_key = row.get("group_key")
        if not isinstance(digest, str) or len(digest) != 64 or not isinstance(group_key, str) or not group_key:
            raise ValueError("Parent cần fingerprint và group key cho từng ảnh.")
        prior_split = digest_splits.setdefault(digest, split)
        if prior_split != split:
            raise ValueError("Parent có ảnh trùng nội dung nằm ở nhiều split.")
        prior_group_split = group_splits.setdefault(group_key, split)
        if prior_group_split != split:
            raise ValueError("Parent có nhóm nguồn bị chia qua nhiều split.")
        image_asset = _safe_parent_asset(parent_root, row.get("exported_image"), "ảnh")
        label_asset = _safe_parent_asset(parent_root, row.get("exported_label"), "label")
        if _sha256(image_asset) != digest:
            raise ValueError("Ảnh parent không khớp SHA-256 manifest: {0}".format(row.get("exported_image")))
        label_bytes = label_asset.read_bytes()
        label_digest = row.get("label_sha256")
        if not isinstance(label_digest, str) or hashlib.sha256(label_bytes).hexdigest() != label_digest:
            raise ValueError("Label parent không khớp SHA-256 manifest: {0}".format(row.get("exported_label")))
        with Image.open(image_asset) as image_file:
            width, height = image_file.size
            image_file.verify()
        if width != row.get("width") or height != row.get("height"):
            raise ValueError("Kích thước ảnh parent không khớp manifest.")
        if split in {"val", "test"} and row.get("status") != "reviewed":
            raise ValueError("Validation/test cố định chỉ được chứa ảnh đã được review.")

        provenance = row.get("annotation_provenance", [])
        if not isinstance(provenance, list):
            raise ValueError("Provenance annotation của parent không hợp lệ.")
        annotations: List[Dict[str, Any]] = []
        has_unreviewed_manual = False
        try:
            label_text = label_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Label parent không phải UTF-8.") from error
        for index, line in enumerate(label_text.splitlines()):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("Label parent không đúng định dạng YOLO.")
            try:
                class_id = int(fields[0])
                x_center, y_center, box_width, box_height = (float(value) for value in fields[1:])
            except ValueError as error:
                raise ValueError("Label parent chứa tọa độ không hợp lệ.") from error
            if class_id not in valid_class_ids or not all(math.isfinite(value) for value in (x_center, y_center, box_width, box_height)):
                raise ValueError("Label parent chứa class hoặc tọa độ không hợp lệ.")
            x_min = (x_center - box_width / 2) * width
            y_min = (y_center - box_height / 2) * height
            x_max = (x_center + box_width / 2) * width
            y_max = (y_center + box_height / 2) * height
            if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
                raise ValueError("Label parent có box vượt ngoài ảnh.")
            source_provenance = provenance[index] if index < len(provenance) and isinstance(provenance[index], dict) else {}
            if source_provenance.get("class_id", class_id) != class_id:
                raise ValueError("Class provenance parent không khớp label YOLO.")
            annotation_source = source_provenance.get("source", "manual")
            annotation_review_status = source_provenance.get("review_status", "unreviewed")
            if annotation_source == "manual" and annotation_review_status != "reviewed":
                has_unreviewed_manual = True
            annotations.append({
                "class_id": class_id,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "source": annotation_source,
                "review_status": annotation_review_status,
                "confidence": source_provenance.get("confidence"),
                "source_model_id": source_provenance.get("source_model_id"),
                "checkpoint_sha256": source_provenance.get("checkpoint_sha256"),
            })
        if len(annotations) != row.get("annotation_count") or len(provenance) != len(annotations):
            raise ValueError("Số label parent không khớp provenance manifest.")
        if split in {"val", "test"} and has_unreviewed_manual:
            raise ValueError("Fixed validation/test của parent chứa box thủ công chưa review; không thể dùng làm tập đánh giá.")
        parent_sample = {
            "path": str(row.get("source_image") or image_asset),
            "parent_asset_path": str(image_asset),
            "parent_label_path": str(label_asset),
            "parent_entry": dict(row),
            "origin_dataset_id": row.get("origin_dataset_id") or dataset_id,
            "source_path": row.get("source_path"),
            "source_id": row.get("source_id"),
            "timestamp_seconds": row.get("timestamp_seconds"),
            "frame_index": row.get("frame_index"),
            "width": width,
            "height": height,
            "status": row.get("status"),
            "annotations": annotations,
            "replay_eligible": not has_unreviewed_manual and (bool(annotations) or row.get("status") == "reviewed"),
            "sha256": digest,
            "group_key": group_key,
            "crop_rect": row.get("crop_rect"),
            "extraction_settings": row.get("extraction_settings"),
            "negative_candidate_provenance": row.get("negative_candidate_provenance", []),
            "split": split,
        }
        by_split[split].append(parent_sample)

    def frozen_hashes(value: Dict[str, Any], split: str) -> set[str]:
        recorded = value.get("image_sha256s")
        actual = sorted(item["sha256"] for item in by_split[split])
        if not isinstance(recorded, list) or sorted(recorded) != actual or not actual:
            raise ValueError("Manifest parent không khớp danh sách ảnh của {0} cố định.".format(split))
        return set(actual)

    frozen_hashes(fixed_validation, "val")
    frozen_hashes(frozen_test, "test")
    if not by_split["train"] or not by_split["val"] or not by_split["test"]:
        raise ValueError("Parent cần có ảnh train, validation và test trước khi làm lineage.")
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "manifest": manifest,
        "dataset_id": dataset_id,
        "version": version,
        "by_split": by_split,
        "digest_splits": digest_splits,
    }


def _balanced_replay_groups(samples: List[Dict[str, Any]], seed: int) -> set[str]:
    class_groups: Dict[int, set[str]] = {}
    negative_groups: set[str] = set()
    for sample in samples:
        classes = {annotation["class_id"] for annotation in sample["annotations"]}
        if not classes:
            negative_groups.add(sample["group_key"])
        for class_id in classes:
            class_groups.setdefault(class_id, set()).add(sample["group_key"])
    if not class_groups:
        return set()
    target = min(len(groups) for groups in class_groups.values())
    selected: set[str] = set()
    for class_id, groups in class_groups.items():
        ordered = sorted(groups, key=lambda key: hashlib.sha256(
            (str(seed) + ":" + str(class_id) + ":" + key).encode("utf-8")
        ).hexdigest())
        selected.update(ordered[:target])
    negative_order = sorted(negative_groups, key=lambda key: hashlib.sha256(
        (str(seed) + ":negative:" + key).encode("utf-8")
    ).hexdigest())
    selected.update(negative_order[:target])
    return selected


def _negative_hash(sample: Dict[str, Any]) -> int:
    path = Path(sample.get("parent_asset_path") or sample["path"])
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không đọc được ảnh negative để kiểm tra ảnh gần giống: {0}".format(path))
    return _dhash(frame)


def _sample_negative_images(
    samples: List[Dict[str, Any]], config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    positives = sum(bool(sample["annotations"]) for sample in samples)
    negatives = [sample for sample in samples if not sample["annotations"]]
    ratio = config["negative_ratio"]
    if ratio == 1:
        target = len(negatives)
        kept = samples
    else:
        target = math.floor(positives * ratio / (1 - ratio)) if positives else 0
        fixed_negatives = [sample for sample in negatives if sample.get("fixed_evaluation")]
        selected = {id(sample) for sample in fixed_negatives}
        quota = max(0, target - len(fixed_negatives))
        recent_hashes = _RecentHashes(threshold=5)
        for sample in sorted(fixed_negatives, key=lambda item: item["sha256"]):
            recent_hashes.add(_negative_hash(sample))

        candidates = [sample for sample in negatives if not sample.get("fixed_evaluation")]
        candidates.sort(key=lambda item: (
            hashlib.sha256((str(config["seed"]) + ":negative:" + item["sha256"]).encode("utf-8")).hexdigest(),
            item["sha256"],
            item["path"],
        ))
        selected_count = 0
        for sample in candidates:
            if selected_count >= quota:
                break
            image_hash = _negative_hash(sample)
            if recent_hashes.contains_near(image_hash):
                continue
            recent_hashes.add(image_hash)
            selected.add(id(sample))
            selected_count += 1
        kept = [sample for sample in samples if sample["annotations"] or id(sample) in selected]

    retained_negative_count = sum(not sample["annotations"] for sample in kept)
    excluded_count = len(negatives) - retained_negative_count
    actual_ratio = retained_negative_count / len(kept) if kept else 0.0
    warnings: List[str] = []
    fixed_negative_count = sum(bool(sample.get("fixed_evaluation")) for sample in negatives)
    if ratio < 1 and fixed_negative_count > target:
        warnings.append(
            "Negative đã khóa trong fixed validation/test vượt quota: tỷ lệ thực tế {0:.1%} cao hơn giới hạn {1:.1%}; các split cố định được giữ nguyên.".format(
                actual_ratio, ratio
            )
        )
    elif excluded_count:
        warnings.append(
            "Áp dụng giới hạn negative {0:.1%} trên dataset cuối: giữ {1}/{2} ảnh đã review, loại {3} ảnh do vượt quota hoặc gần giống (dHash ≤5).".format(
                ratio, retained_negative_count, len(negatives), excluded_count
            )
        )
    return kept, warnings, {
        "negative_ratio_requested": ratio,
        "negative_ratio_actual": actual_ratio,
        "negative_sample_target_count": target,
        "negative_samples_available_count": len(negatives),
        "negative_samples_excluded_count": excluded_count,
    }


def _compose_lineage_samples(
    current_samples: List[Dict[str, Any]],
    parent: Dict[str, Any],
    config: Dict[str, Any],
    replay_strategy: str,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    parent_splits = parent["by_split"]
    fixed_val = parent_splits["val"]
    frozen_test = parent_splits["test"]
    all_historical = parent_splits["train"]
    historical = [sample for sample in all_historical if sample.get("replay_eligible", True)]
    fixed_groups = {sample["group_key"] for sample in fixed_val + frozen_test}
    historical_groups = {sample["group_key"] for sample in historical}
    if replay_strategy == "all":
        replay_groups = historical_groups
    elif replay_strategy == "balanced":
        replay_groups = _balanced_replay_groups(historical, config["seed"])
    else:
        replay_groups = set()

    warnings: List[str] = []
    if replay_strategy == "balanced" and not replay_groups:
        warnings.append("Parent không có nhóm positive đủ class để chọn replay cân bằng; chỉ dùng dữ liệu mới và các split cố định.")
    excluded_parent_train_count = len(all_historical) - len(historical)
    if excluded_parent_train_count:
        warnings.append(
            "Đã loại {0} ảnh train trong parent vì có nhãn thủ công chưa review hoặc ảnh âm tính chưa review.".format(
                excluded_parent_train_count
            )
        )
    by_digest: Dict[str, List[Dict[str, Any]]] = {}
    for sample in current_samples:
        by_digest.setdefault(sample["sha256"], []).append(sample)
    parent_digest_split = {
        digest: split for digest, split in parent["digest_splits"].items() if split != "train"
    }
    parent_digest_split.update({sample["sha256"]: "train" for sample in historical})
    used_current_paths: set[str] = set()
    result: List[Dict[str, Any]] = []

    for split, fixed_samples in (("val", fixed_val), ("test", frozen_test)):
        for sample in fixed_samples:
            result.append({**sample, "split": split, "fixed_evaluation": True})

    replayed_image_count = 0
    for old_sample in historical:
        if old_sample["group_key"] not in replay_groups:
            continue
        current_matches = by_digest.get(old_sample["sha256"], [])
        replacement = next((item for item in current_matches if item["path"] not in used_current_paths), None)
        if replacement is not None:
            used_current_paths.add(replacement["path"])
            result.append({**replacement, "split": "train", "origin_dataset_id": old_sample["origin_dataset_id"]})
        else:
            result.append({**old_sample, "split": "train"})
        replayed_image_count += 1

    new_samples: List[Dict[str, Any]] = []
    for sample in current_samples:
        if sample["path"] in used_current_paths:
            continue
        if sample["group_key"] in fixed_groups:
            continue
        old_split = parent_digest_split.get(sample["sha256"])
        if old_split is not None:
            if old_split == "train" and sample["group_key"] in replay_groups:
                result.append({**sample, "split": "train"})
                replayed_image_count += 1
            continue
        if sample["group_key"] in historical_groups:
            if sample["group_key"] in replay_groups:
                result.append({**sample, "split": "train"})
                replayed_image_count += 1
            continue
        new_samples.append(sample)

    new_split_config = {
        **config,
        "ratios": {
            "train": config["ratios"]["train"],
            "val": config["ratios"]["val"],
            "test": 0.0,
        },
    }
    assignments, split_warnings = _split_groups(new_samples, new_split_config)
    warnings.extend(split_warnings)
    new_image_count = 0
    for sample in new_samples:
        split = assignments.get(sample["path"], "train")
        result.append({**sample, "split": split, "origin_dataset_id": None})
        new_image_count += 1

    return result, warnings, {
        "new_image_count": new_image_count,
        "replayed_image_count": replayed_image_count,
        "unreviewed_parent_train_excluded_count": excluded_parent_train_count,
        "fixed_validation_count": len(fixed_val),
        "frozen_test_count": len(frozen_test),
    }


def _source_key(image: Dict[str, Any]) -> Optional[str]:
    source_id = image.get("source_id")
    if isinstance(source_id, str) and source_id.strip():
        return "source-id:" + source_id.strip()
    source_path = image.get("source_path")
    if isinstance(source_path, str) and source_path.strip():
        try:
            return "source-path:" + str(Path(source_path).expanduser().resolve())
        except OSError:
            return "source-path:" + os.path.normcase(os.path.abspath(source_path))
    return None


def _source_fps(project: Dict[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for source in project.get("sources", []):
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            continue
        fps = source.get("fps")
        if isinstance(fps, (int, float)) and not isinstance(fps, bool) and math.isfinite(fps) and fps > 0:
            try:
                result[str(Path(source["path"]).expanduser().resolve())] = float(fps)
            except OSError:
                continue
    return result


def _frame_time(image: Dict[str, Any], fps_by_path: Dict[str, float]) -> Optional[float]:
    timestamp = image.get("timestamp_seconds")
    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool) and math.isfinite(timestamp) and timestamp >= 0:
        return float(timestamp)
    source_path = image.get("source_path")
    frame_index = image.get("frame_index")
    if not isinstance(source_path, str) or isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        return None
    try:
        fps = fps_by_path.get(str(Path(source_path).expanduser().resolve()))
    except OSError:
        return None
    return frame_index / fps if fps else None


def _split_groups(samples: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    """Keep source/segment groups and byte-identical images in the same split."""
    parents = list(range(len(samples)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    group_owner: Dict[str, int] = {}
    digest_owner: Dict[str, int] = {}
    for index, sample in enumerate(samples):
        group_key = sample["group_key"]
        if group_key in group_owner:
            union(index, group_owner[group_key])
        else:
            group_owner[group_key] = index
        digest = sample["sha256"]
        if digest in digest_owner:
            union(index, digest_owner[digest])
        else:
            digest_owner[digest] = index

    combined: Dict[int, Dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        root = find(index)
        group = combined.setdefault(root, {"keys": set(), "sample_indexes": []})
        group["keys"].add(sample["group_key"])
        group["sample_indexes"].append(index)

    groups = list(combined.values())
    seed = config["seed"]
    groups.sort(key=lambda group: hashlib.sha256(
        (str(seed) + ":" + "|".join(sorted(group["keys"]))).encode("utf-8")
    ).hexdigest())
    ratios = config["ratios"]
    active_splits = [name for name in SPLITS if ratios[name] > 0]
    warnings: List[str] = []
    assigned: Dict[str, str] = {}
    split_counts = {name: 0 for name in SPLITS}
    total = len(samples)

    if len(groups) < len(active_splits):
        warnings.append(
            "Không đủ nhóm độc lập để tạo mọi split đã chọn; các nhóm đầu được cấp cho train rồi val, còn split thiếu cần thêm nguồn độc lập."
        )
        for position, group in enumerate(groups):
            destination = active_splits[position]
            for index in group["sample_indexes"]:
                assigned[samples[index]["path"]] = destination
                split_counts[destination] += 1
    else:
        targets = {name: total * ratios[name] for name in SPLITS}
        for position, group in enumerate(groups):
            size = len(group["sample_indexes"])
            if position < len(active_splits):
                destination = active_splits[position]
            else:
                def score(candidate: str) -> float:
                    return sum(
                        ((split_counts[name] + (size if name == candidate else 0) - targets[name]) ** 2)
                        / max(targets[name], 1.0)
                        for name in active_splits
                    )

                destination = min(active_splits, key=score)
            for index in group["sample_indexes"]:
                assigned[samples[index]["path"]] = destination
            split_counts[destination] += size

    return assigned, warnings


def _analyze(
    path_value: str,
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    segment_seconds: float,
    parent_manifest_path: Optional[str] = None,
    replay_strategy: str = "all",
    freeze_evaluation_set: bool = False,
    negative_ratio: float = 0.1,
) -> Dict[str, Any]:
    config = _config(split_mode, train_ratio, val_ratio, test_ratio, seed, segment_seconds, negative_ratio)
    project = load_project(path_value)
    class_names = project.get("class_names", [])
    images = project.get("images", [])
    errors: List[str] = []
    warnings: List[str] = []
    if replay_strategy not in {"all", "balanced", "new"}:
        errors.append("Replay phải là toàn bộ ảnh train cũ, tập cũ cân bằng hoặc chỉ ảnh mới.")
        replay_strategy = "all"
    if not isinstance(freeze_evaluation_set, bool):
        errors.append("Cờ đóng băng validation/test không hợp lệ.")
        freeze_evaluation_set = False
    if not isinstance(class_names, list) or not class_names or any(not isinstance(name, str) or not name.strip() for name in class_names):
        errors.append("Dự án thiếu danh sách class hợp lệ.")
        class_names = []
    elif len({name.strip().casefold() for name in class_names}) != len(class_names):
        errors.append("Tên class bị trùng; hãy sửa mapping class trước khi export.")
    if not isinstance(images, list) or not images:
        errors.append("Dataset đang trống; hãy trích xuất ảnh trước.")
        images = []

    valid_class_ids = set(range(len(class_names)))
    fps_by_path = _source_fps(project)
    samples: List[Dict[str, Any]] = []
    candidate_negative_count = 0
    basename_groups: Dict[str, List[str]] = {}
    exact_hashes: Dict[str, List[str]] = {}
    class_counts = [0 for _ in class_names]
    reviewed_negative_count = 0
    unreviewed_annotation_excluded_count = 0
    positive_count = 0
    annotation_count = 0
    pending_suggestion_count = 0
    pending_negative_candidate_count = 0
    missing_source_metadata = 0

    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get("path"), str) or not image["path"]:
            errors.append("Dự án có bản ghi ảnh thiếu đường dẫn.")
            continue
        try:
            image_path = Path(image["path"]).expanduser().resolve(strict=True)
        except OSError:
            errors.append("Không tìm thấy ảnh trong dự án: {0}".format(image["path"]))
            continue
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            errors.append("Đường dẫn ảnh không phải tệp ảnh được hỗ trợ: {0}".format(image_path))
            continue
        try:
            with Image.open(image_path) as decoded:
                actual_width, actual_height = decoded.size
                decoded.verify()
            if actual_width <= 0 or actual_height <= 0:
                raise ValueError("Kích thước ảnh không hợp lệ.")
        except (OSError, ValueError, UnidentifiedImageError) as error:
            errors.append("Ảnh hỏng hoặc không đọc được: {0} ({1})".format(image_path, error))
            continue

        stored_width, stored_height = image.get("width"), image.get("height")
        if stored_width != actual_width or stored_height != actual_height:
            errors.append(
                "Kích thước ảnh đã đổi kể từ lúc lưu dự án: {0} (dự án {1}×{2}, tệp {3}×{4}).".format(
                    image_path, stored_width, stored_height, actual_width, actual_height
                )
            )
            continue

        annotations = image.get("annotations", [])
        if not isinstance(annotations, list):
            errors.append("Danh sách nhãn không hợp lệ: {0}".format(image_path))
            continue
        cleaned_annotations = []
        image_invalid = False
        has_pending_ai = False
        has_unreviewed_manual = False
        image_pending_ai_count = 0
        for annotation in annotations:
            if not isinstance(annotation, dict):
                errors.append("Nhãn không đúng cấu trúc trong ảnh: {0}".format(image_path))
                image_invalid = True
                continue
            class_id = annotation.get("class_id")
            if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id not in valid_class_ids:
                errors.append("Class ID không tồn tại trong mapping: {0}".format(image_path))
                image_invalid = True
                continue
            try:
                coords = [annotation[key] for key in ("x_min", "y_min", "x_max", "y_max")]
                if any(isinstance(value, bool) for value in coords):
                    raise ValueError("tọa độ boolean")
                x_min, y_min, x_max, y_max = (float(value) for value in coords)
            except (KeyError, TypeError, ValueError):
                errors.append("Tọa độ box không hợp lệ: {0}".format(image_path))
                image_invalid = True
                continue
            if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
                errors.append("Tọa độ box không hữu hạn: {0}".format(image_path))
                image_invalid = True
                continue
            if not (0 <= x_min < x_max <= actual_width and 0 <= y_min < y_max <= actual_height):
                errors.append("Box có diện tích bằng 0 hoặc vượt ảnh: {0}".format(image_path))
                image_invalid = True
                continue
            source = annotation.get("source", "manual")
            review_status = annotation.get("review_status", "unreviewed")
            if source not in {"manual", "ai"}:
                errors.append("Nguồn annotation không hợp lệ: {0}".format(image_path))
                image_invalid = True
                continue
            if source == "manual" and review_status not in {"unreviewed", "reviewed"}:
                errors.append("Trạng thái review của nhãn thủ công không hợp lệ: {0}".format(image_path))
                image_invalid = True
                continue
            if source == "ai" and review_status not in {"needs_review", "reviewed", "rejected"}:
                errors.append("Trạng thái review của gợi ý AI không hợp lệ: {0}".format(image_path))
                image_invalid = True
                continue
            confidence = annotation.get("confidence")
            if confidence is not None:
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    errors.append("Confidence của annotation không hợp lệ: {0}".format(image_path))
                    image_invalid = True
                    continue
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    errors.append("Confidence của annotation không hợp lệ: {0}".format(image_path))
                    image_invalid = True
                    continue
            source_model_id = annotation.get("source_model_id")
            checkpoint_sha256 = annotation.get("checkpoint_sha256")
            if source == "manual" and (source_model_id is not None or checkpoint_sha256 is not None):
                errors.append("Nhãn thủ công không được mang provenance checkpoint: {0}".format(image_path))
                image_invalid = True
                continue
            if source == "ai" and review_status == "reviewed" and (
                not isinstance(source_model_id, str)
                or not source_model_id
                or not isinstance(checkpoint_sha256, str)
                or len(checkpoint_sha256) != 64
                or any(character not in "0123456789abcdef" for character in checkpoint_sha256.casefold())
            ):
                errors.append("Gợi ý AI đã duyệt thiếu provenance checkpoint: {0}".format(image_path))
                image_invalid = True
                continue
            if source == "ai" and review_status == "needs_review":
                has_pending_ai = True
                image_pending_ai_count += 1
            if source == "manual" and review_status == "unreviewed":
                has_unreviewed_manual = True
            if source == "ai" and review_status != "reviewed":
                continue
            cleaned_annotations.append({
                "class_id": class_id,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "source": source,
                "review_status": review_status,
                "confidence": confidence,
                "source_model_id": source_model_id,
                "checkpoint_sha256": checkpoint_sha256,
            })
        if image_invalid:
            continue
        pending_suggestion_count += image_pending_ai_count

        negative_candidates = image.get("negative_candidates", [])
        if not isinstance(negative_candidates, list):
            errors.append("Danh sách candidate âm tính không hợp lệ: {0}".format(image_path))
            continue
        pending_candidates = [
            item for item in negative_candidates
            if isinstance(item, dict) and item.get("review_status") == "needs_review"
        ]
        confirmed_candidates = [
            item for item in negative_candidates
            if isinstance(item, dict) and item.get("review_status") == "confirmed_negative"
        ]
        if has_unreviewed_manual:
            unreviewed_annotation_excluded_count += 1
        if pending_candidates:
            pending_negative_candidate_count += len(pending_candidates)
            candidate_negative_count += 1
            continue

        status = image.get("status", "ready")
        if status not in {"ready", "reviewed"}:
            errors.append("Trạng thái review của ảnh không hợp lệ: {0}".format(image_path))
            continue
        if has_unreviewed_manual:
            continue
        if cleaned_annotations:
            positive_count += 1
            annotation_count += len(cleaned_annotations)
            for annotation in cleaned_annotations:
                class_counts[annotation["class_id"]] += 1
        elif status == "reviewed" and not has_pending_ai:
            reviewed_negative_count += 1
        else:
            candidate_negative_count += 1
            continue

        try:
            digest = _sha256(image_path)
        except OSError as error:
            errors.append("Không đọc được ảnh để kiểm tra integrity: {0} ({1})".format(image_path, error))
            continue
        if not cleaned_annotations and status == "reviewed" and any(
            candidate.get("image_fingerprint") != digest
            or not isinstance(candidate.get("source_model_id"), str)
            or not candidate.get("source_model_id")
            or not isinstance(candidate.get("checkpoint_sha256"), str)
            or len(candidate["checkpoint_sha256"]) != 64
            for candidate in confirmed_candidates
        ):
            errors.append("Ảnh âm tính đã thay đổi hoặc thiếu provenance candidate: {0}".format(image_path))
            continue
        source_key = _source_key(image)
        if source_key is None:
            missing_source_metadata += 1
            source_key = "image:" + str(image_path)
        if config["split_mode"] == "segment" and _source_key(image) is not None:
            timestamp = _frame_time(image, fps_by_path)
            if timestamp is not None:
                segment = int(timestamp // config["segment_seconds"])
                source_key = "segment:{0}:{1}".format(source_key, segment)
        basename_groups.setdefault(image_path.name.casefold(), []).append(str(image_path))
        exact_hashes.setdefault(digest, []).append(str(image_path))
        samples.append({
            "path": str(image_path),
            "source_path": image.get("source_path"),
            "source_id": image.get("source_id"),
            "source_key": _source_key(image),
            "timestamp_seconds": image.get("timestamp_seconds"),
            "frame_index": image.get("frame_index"),
            "width": actual_width,
            "height": actual_height,
            "status": status,
            "annotations": cleaned_annotations,
            "sha256": digest,
            "group_key": source_key,
            "crop_rect": image.get("crop_rect"),
            "extraction_settings": image.get("extraction_settings"),
            "negative_candidate_provenance": [
                {
                    "candidate_id": candidate.get("id"),
                    "source_model_id": candidate.get("source_model_id"),
                    "checkpoint_sha256": candidate.get("checkpoint_sha256"),
                    "image_fingerprint": candidate.get("image_fingerprint"),
                    "review_status": candidate.get("review_status"),
                }
                for candidate in confirmed_candidates
            ] if not cleaned_annotations else [],
        })

    if candidate_negative_count:
        warnings.append(
            "Đã loại {0} ảnh trống chưa review hoặc candidate âm tính chưa được xác nhận; ảnh AI không phát hiện chỉ vào dataset sau khi người dùng xác nhận.".format(candidate_negative_count)
        )
    if unreviewed_annotation_excluded_count:
        warnings.append(
            "Đã loại {0} ảnh có box thủ công chưa được review; hãy đánh dấu ảnh đã review để đưa nhãn vào dataset.".format(
                unreviewed_annotation_excluded_count
            )
        )
    if pending_suggestion_count:
        warnings.append(
            "Đã loại {0} gợi ý AI chưa duyệt khỏi nhãn YOLO mặc định; ảnh chỉ có gợi ý đang chờ duyệt không được dùng làm negative.".format(pending_suggestion_count)
        )
    if missing_source_metadata:
        warnings.append(
            "{0} ảnh thiếu source video; split theo source không thể bảo đảm chúng cùng phiên thu thập.".format(missing_source_metadata)
        )
    duplicate_basename_groups = [paths for paths in basename_groups.values() if len(paths) > 1]
    if duplicate_basename_groups:
        warnings.append(
            "Có {0} nhóm trùng tên file; exporter sẽ thêm hậu tố để tránh ghi đè.".format(len(duplicate_basename_groups))
        )
    exact_duplicate_groups = [paths for paths in exact_hashes.values() if len(paths) > 1]
    exact_duplicate_count = sum(len(paths) - 1 for paths in exact_duplicate_groups)
    if exact_duplicate_count:
        warnings.append(
            "Phát hiện {0} ảnh trùng nội dung byte; các bản trùng được giữ cùng split để tránh rò rỉ.".format(exact_duplicate_count)
        )

    parent = None
    parent_value = parent_manifest_path.strip() if isinstance(parent_manifest_path, str) else ""
    if parent_manifest_path is not None and not isinstance(parent_manifest_path, str):
        errors.append("Đường dẫn parent manifest không hợp lệ.")
    if parent_value:
        try:
            parent = _read_parent_dataset(parent_value, class_names, config)
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(str(error))

    lineage_counts = {
        "new_image_count": 0,
        "replayed_image_count": 0,
        "unreviewed_parent_train_excluded_count": 0,
        "fixed_validation_count": 0,
        "frozen_test_count": 0,
    }
    if parent is not None:
        if config["ratios"]["test"] != 0:
            errors.append("Khi dùng parent, đặt tỷ lệ test của dữ liệu mới bằng 0 để giữ nguyên test set đã đóng băng.")
        if freeze_evaluation_set:
            errors.append("Bộ val/test của parent đã đóng băng; không thể tạo bộ mới trong cùng lượt xuất.")
        samples, lineage_warnings, lineage_counts = _compose_lineage_samples(samples, parent, config, replay_strategy)
        warnings.extend(lineage_warnings)
    else:
        split_assignments, split_warnings = _split_groups(samples, config)
        warnings.extend(split_warnings)
        for sample in samples:
            sample["split"] = split_assignments.get(sample["path"], "train")
        if freeze_evaluation_set:
            if not samples or not config["ratios"]["val"] or not config["ratios"]["test"]:
                errors.append("Đóng băng cần có ảnh trong cả validation và test.")
            if (
                not any(sample["split"] == "val" for sample in samples)
                or not any(sample["split"] == "test" for sample in samples)
            ):
                errors.append("Đóng băng cần ít nhất một ảnh trong cả validation và test sau khi chia nhóm.")
            unreviewed = [sample for sample in samples if sample["split"] in {"val", "test"} and sample["status"] != "reviewed"]
            if unreviewed:
                errors.append("Chỉ đóng băng được val/test khi mọi ảnh đã được người dùng review ({0} ảnh còn lại).".format(len(unreviewed)))
            lineage_counts["fixed_validation_count"] = sum(sample["split"] == "val" for sample in samples)
            lineage_counts["frozen_test_count"] = sum(sample["split"] == "test" for sample in samples)

    samples, negative_warnings, negative_summary = _sample_negative_images(samples, config)
    warnings.extend(negative_warnings)
    if parent is not None:
        lineage_counts["new_image_count"] = sum(sample.get("origin_dataset_id") is None for sample in samples)
        lineage_counts["replayed_image_count"] = sum(
            sample["split"] == "train" and sample.get("origin_dataset_id") is not None for sample in samples
        )
        lineage_counts["fixed_validation_count"] = sum(
            bool(sample.get("fixed_evaluation")) and sample["split"] == "val" for sample in samples
        )
        lineage_counts["frozen_test_count"] = sum(
            bool(sample.get("fixed_evaluation")) and sample["split"] == "test" for sample in samples
        )
    elif freeze_evaluation_set:
        lineage_counts["fixed_validation_count"] = sum(sample["split"] == "val" for sample in samples)
        lineage_counts["frozen_test_count"] = sum(sample["split"] == "test" for sample in samples)

    split_counts = {name: 0 for name in SPLITS}
    split_group_keys: Dict[str, set[str]] = {name: set() for name in SPLITS}
    for sample in samples:
        split = sample["split"]
        split_counts[split] += 1
        split_group_keys[split].add(sample["group_key"])
    if split_counts["train"] == 0 and samples:
        errors.append("Split train không có ảnh; hãy đổi seed hoặc cấu hình split.")
    for split in (SPLITS if parent is None else ("train", "val")):
        if config["ratios"][split] > 0 and split_counts[split] == 0:
            warnings.append("Split {0} đang trống dù tỷ lệ được đặt lớn hơn 0%.".format(split))
    if samples and parent is None:
        for split in SPLITS:
            actual_ratio = split_counts[split] / len(samples)
            if abs(actual_ratio - config["ratios"][split]) > 0.15:
                warnings.append(
                    "Tỷ lệ {0} thực tế là {1:.0%}, khác đáng kể mục tiêu {2:.0%} do chia theo nhóm chống rò rỉ.".format(
                        split, actual_ratio, config["ratios"][split]
                    )
                )

    class_counts = [0 for _ in class_names]
    positive_count = 0
    reviewed_negative_count = 0
    annotation_count = 0
    exact_hash_counts: Dict[str, int] = {}
    for sample in samples:
        exact_hash_counts[sample["sha256"]] = exact_hash_counts.get(sample["sha256"], 0) + 1
        if sample["annotations"]:
            positive_count += 1
            annotation_count += len(sample["annotations"])
            for annotation in sample["annotations"]:
                class_counts[annotation["class_id"]] += 1
        elif sample["status"] == "reviewed":
            reviewed_negative_count += 1
    exact_duplicate_count = sum(count - 1 for count in exact_hash_counts.values() if count > 1)
    class_distribution = [
        {"class_id": class_id, "name": name, "box_count": class_counts[class_id]}
        for class_id, name in enumerate(class_names)
    ]
    positive_class_counts = [count for count in class_counts if count > 0]
    if any(count == 0 for count in class_counts):
        warnings.append("Có class chưa xuất hiện trong annotation: {0}.".format(
            ", ".join(class_names[index] for index, count in enumerate(class_counts) if count == 0)
        ))
    elif len(positive_class_counts) > 1 and max(positive_class_counts) > 10 * min(positive_class_counts):
        warnings.append("Phân bố class lệch hơn 10:1; hãy kiểm tra class hiếm trước khi huấn luyện.")
    if not samples and images and parent is None:
        errors.append("Không có ảnh hợp lệ để export sau khi loại negative chưa review.")
    if not samples:
        errors.append("Dataset không có ảnh sau khi áp dụng lineage/replay.")

    lineage = {
        "parent_dataset_id": parent["dataset_id"] if parent else None,
        "parent_version": parent["version"] if parent else None,
        "parent_manifest_path": str(parent["manifest_path"]) if parent else None,
        "parent_manifest_sha256": parent["manifest_sha256"] if parent else None,
        "replay_strategy": replay_strategy if parent else "none",
        "freeze_evaluation_set": bool(freeze_evaluation_set and parent is None),
        **lineage_counts,
    }

    summary = {
        "valid": not errors,
        "project_name": project.get("name", "Dự án FrameForge"),
        "image_count": len(images),
        "included_image_count": len(samples),
        "new_image_count": lineage_counts["new_image_count"] if parent else len(samples),
        "replayed_image_count": lineage_counts["replayed_image_count"],
        "fixed_validation_count": lineage_counts["fixed_validation_count"],
        "frozen_test_count": lineage_counts["frozen_test_count"],
        "positive_image_count": positive_count,
        "reviewed_negative_count": reviewed_negative_count,
        "unreviewed_annotation_excluded_count": unreviewed_annotation_excluded_count,
        "unreviewed_parent_train_excluded_count": lineage_counts["unreviewed_parent_train_excluded_count"],
        "candidate_negative_excluded_count": candidate_negative_count,
        "annotation_count": annotation_count,
        "pending_suggestion_count": pending_suggestion_count,
        "pending_negative_candidate_count": pending_negative_candidate_count,
        "class_distribution": class_distribution,
        "duplicate_basename_groups": len(duplicate_basename_groups),
        "exact_duplicate_count": exact_duplicate_count,
        "split_counts": split_counts,
        "split_group_counts": {name: len(values) for name, values in split_group_keys.items()},
        "split_mode": config["split_mode"],
        "ratios": config["ratios"],
        "seed": config["seed"],
        "segment_seconds": config["segment_seconds"],
        **negative_summary,
        "errors": errors,
        "warnings": warnings,
    }
    return {"summary": summary, "samples": samples, "project": project, "config": config, "lineage": lineage, "parent": parent}


def validate_dataset(
    path_value: str,
    split_mode: str = "source",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    segment_seconds: float = 30.0,
    parent_manifest_path: Optional[str] = None,
    replay_strategy: str = "all",
    freeze_evaluation_set: bool = False,
    negative_ratio: float = 0.1,
) -> Dict[str, Any]:
    """Validate a saved project and return a stable, source-grouped split preview."""
    return _analyze(
        path_value, split_mode, train_ratio, val_ratio, test_ratio, seed, segment_seconds,
        parent_manifest_path, replay_strategy, freeze_evaluation_set, negative_ratio,
    )["summary"]


def _safe_filename(path: Path, digest: str, reserved: set[str]) -> str:
    stem = "".join(character if character.isalnum() or character in "-_ ." else "_" for character in path.stem)
    stem = re.sub(r"\s+", "_", stem).strip(" ._") or "image"
    extension = path.suffix.lower()
    name = stem + extension
    if name.casefold() in reserved:
        name = "{0}--{1}{2}".format(stem, digest[:8], extension)
    counter = 2
    while name.casefold() in reserved:
        name = "{0}--{1}-{2}{3}".format(stem, digest[:8], counter, extension)
        counter += 1
    reserved.add(name.casefold())
    return name


def _project_annotation_revision(project: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(project.get("class_names", []), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    for image in sorted(project.get("images", []), key=lambda item: str(item.get("path", ""))):
        value = {
            "path": image.get("path"),
            "status": image.get("status"),
            "annotations": image.get("annotations", []),
            "negative_candidates": image.get("negative_candidates", []),
        }
        digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def export_yolo_dataset(
    path_value: str,
    output_dir: str,
    split_mode: str = "source",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    segment_seconds: float = 30.0,
    parent_manifest_path: Optional[str] = None,
    replay_strategy: str = "all",
    freeze_evaluation_set: bool = False,
    negative_ratio: float = 0.1,
) -> Dict[str, Any]:
    """Create a new immutable YOLO dataset version without modifying source images."""
    analysis = _analyze(
        path_value, split_mode, train_ratio, val_ratio, test_ratio, seed, segment_seconds,
        parent_manifest_path, replay_strategy, freeze_evaluation_set, negative_ratio,
    )
    summary = analysis["summary"]
    if not summary["valid"]:
        raise ValueError("Dataset chưa hợp lệ:\n- " + "\n- ".join(summary["errors"]))
    destination_root = Path(output_dir).expanduser().resolve(strict=True)
    if not destination_root.is_dir():
        raise ValueError("Thư mục export dataset không hợp lệ.")

    project = analysis["project"]
    samples = analysis["samples"]
    config = analysis["config"]
    lineage = analysis["lineage"]
    parent = analysis["parent"]
    dataset_id = str(uuid.uuid4())
    version = parent["version"] + 1 if parent else 1
    created_at = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = "frameforge_dataset_{0}_{1}".format(stamp, dataset_id[:8])
    final_root = destination_root / dataset_name
    temporary_root = destination_root / ("." + dataset_name + ".tmp")
    reserved_names: set[str] = set()
    manifest_images = []

    try:
        temporary_root.mkdir(mode=0o700)
        for split in SPLITS:
            (temporary_root / "images" / split).mkdir(parents=True)
            (temporary_root / "labels" / split).mkdir(parents=True)

        for sample in samples:
            source = Path(sample.get("parent_asset_path") or sample["path"])
            split = sample["split"]
            filename = _safe_filename(source, sample["sha256"], reserved_names)
            image_relative = Path("images") / split / filename
            label_relative = Path("labels") / split / (Path(filename).stem + ".txt")
            image_destination = temporary_root / image_relative
            shutil.copy2(source, image_destination)
            if _sha256(image_destination) != sample["sha256"]:
                raise ValueError("Ảnh nguồn thay đổi trong lúc export: {0}".format(source))

            parent_entry = sample.get("parent_entry")
            if isinstance(parent_entry, dict):
                parent_label = Path(sample["parent_label_path"])
                label_bytes = parent_label.read_bytes()
                if hashlib.sha256(label_bytes).hexdigest() != parent_entry.get("label_sha256"):
                    raise ValueError("Label parent thay đổi trong lúc export: {0}".format(parent_label))
                (temporary_root / label_relative).write_bytes(label_bytes)
                manifest_images.append({
                    **parent_entry,
                    "exported_image": image_relative.as_posix(),
                    "exported_label": label_relative.as_posix(),
                    "split": split,
                })
                continue

            label_lines = []
            for annotation in sample["annotations"]:
                width = sample["width"]
                height = sample["height"]
                x_min, y_min = annotation["x_min"], annotation["y_min"]
                x_max, y_max = annotation["x_max"], annotation["y_max"]
                x_center = ((x_min + x_max) / 2) / width
                y_center = ((y_min + y_max) / 2) / height
                box_width = (x_max - x_min) / width
                box_height = (y_max - y_min) / height
                label_lines.append("{0} {1:.8f} {2:.8f} {3:.8f} {4:.8f}".format(
                    annotation["class_id"], x_center, y_center, box_width, box_height
                ))
            label_text = "\n".join(label_lines) + ("\n" if label_lines else "")
            (temporary_root / label_relative).write_text(label_text, encoding="utf-8")
            manifest_images.append({
                "source_image": sample["path"],
                "exported_image": image_relative.as_posix(),
                "exported_label": label_relative.as_posix(),
                "sha256": sample["sha256"],
                "label_sha256": hashlib.sha256(label_text.encode("utf-8")).hexdigest(),
                "width": sample["width"],
                "height": sample["height"],
                "source_path": sample["source_path"],
                "source_id": sample["source_id"],
                "frame_index": sample["frame_index"],
                "timestamp_seconds": sample["timestamp_seconds"],
                "crop_rect": sample["crop_rect"],
                "extraction_settings": sample["extraction_settings"],
                "status": sample["status"],
                "split": split,
                "group_key": sample["group_key"],
                "origin_dataset_id": sample.get("origin_dataset_id") or dataset_id,
                "annotation_count": len(sample["annotations"]),
                "annotation_provenance": [
                    {
                        "class_id": annotation["class_id"],
                        "source": annotation["source"],
                        "review_status": annotation["review_status"],
                        "confidence": annotation["confidence"],
                        "source_model_id": annotation["source_model_id"],
                        "checkpoint_sha256": annotation["checkpoint_sha256"],
                    }
                    for annotation in sample["annotations"]
                ],
                "negative_candidate_provenance": sample["negative_candidate_provenance"],
            })

        names = {index: name for index, name in enumerate(project["class_names"])}
        dataset_yaml = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": names,
        }
        (temporary_root / "data.yaml").write_text(
            yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        if parent is not None:
            fixed_validation_set = parent["manifest"]["fixed_validation_set"]
            frozen_test_set = parent["manifest"]["frozen_test_set"]
        elif freeze_evaluation_set:
            fixed_validation_set = {
                "enabled": True,
                "set_id": dataset_id + ":validation",
                "dataset_id": dataset_id,
                "image_sha256s": sorted(sample["sha256"] for sample in samples if sample["split"] == "val"),
                "created_at": created_at,
            }
            frozen_test_set = {
                "enabled": True,
                "set_id": dataset_id + ":test",
                "dataset_id": dataset_id,
                "image_sha256s": sorted(sample["sha256"] for sample in samples if sample["split"] == "test"),
                "created_at": created_at,
            }
        else:
            fixed_validation_set = {"enabled": False, "image_sha256s": []}
            frozen_test_set = {"enabled": False, "image_sha256s": []}
        manifest = {
            "format": "FrameForge YOLO dataset",
            "dataset_id": dataset_id,
            "version": version,
            "parent_manifest_path": lineage["parent_manifest_path"],
            "parent_dataset_id": lineage["parent_dataset_id"],
            "parent_version": lineage["parent_version"],
            "parent_manifest_sha256": lineage["parent_manifest_sha256"],
            "replay_strategy": lineage["replay_strategy"],
            "lineage": lineage,
            "annotation_revision": _project_annotation_revision(project),
            "created_at": created_at,
            "project_id": project.get("id"),
            "project_name": project.get("name"),
            "class_mapping": names,
            "fixed_validation_set": fixed_validation_set,
            "frozen_test_set": frozen_test_set,
            "source_files": sorted({sample["source_path"] for sample in samples if sample["source_path"]}),
            "split": {
                "algorithm": "stable-seeded-group-greedy-v1",
                "mode": config["split_mode"],
                "assignment_scope": "new_project_images" if parent else "all_project_images",
                "ratios": config["ratios"],
                "seed": config["seed"],
                "segment_seconds": config["segment_seconds"],
                "negative_ratio": config["negative_ratio"],
                "counts": summary["split_counts"],
                "groups": summary["split_group_counts"],
            },
            "summary": {
                key: summary[key]
                for key in (
                    "image_count", "included_image_count", "new_image_count", "replayed_image_count",
                    "fixed_validation_count", "frozen_test_count", "positive_image_count",
                    "reviewed_negative_count", "unreviewed_annotation_excluded_count",
                    "unreviewed_parent_train_excluded_count", "candidate_negative_excluded_count",
                    "negative_ratio_requested", "negative_ratio_actual", "negative_sample_target_count",
                    "negative_samples_available_count", "negative_samples_excluded_count",
                    "annotation_count", "pending_suggestion_count", "class_distribution", "exact_duplicate_count",
                )
            },
            "warnings": summary["warnings"],
            "images": manifest_images,
        }
        (temporary_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_root.rename(final_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    return {
        "dataset_id": dataset_id,
        "version": version,
        "parent_dataset_id": lineage["parent_dataset_id"],
        "replay_strategy": lineage["replay_strategy"],
        "path": str(final_root),
        "data_yaml": str(final_root / "data.yaml"),
        "image_count": summary["included_image_count"],
        "new_image_count": summary["new_image_count"],
        "replayed_image_count": summary["replayed_image_count"],
        "fixed_validation_count": summary["fixed_validation_count"],
        "frozen_test_count": summary["frozen_test_count"],
        "split_counts": summary["split_counts"],
        "warnings": summary["warnings"],
    }
