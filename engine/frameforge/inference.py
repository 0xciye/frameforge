"""YOLO11n inference on extracted images using verified FrameForge checkpoints."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from frameforge.training import IMAGE_EXTENSIONS, training_capabilities, verified_checkpoint_record
from frameforge.project import _save_ai_suggestions_batch, project_images_for_inference


MAX_INFERENCE_IMAGES = 500
INFERENCE_BATCH_SIZE = 16
MAX_DETECTIONS_PER_IMAGE = 300
CLIPPED_BOX_EDGE_MARGIN_PX = 1.0
PROJECT_PERSIST_BATCH_SIZE = 64
MAX_PROJECT_INFERENCE_IMAGES = 100_000


def _names_in_order(value: Any) -> List[str]:
    if isinstance(value, dict):
        try:
            ordered = sorted(value.items(), key=lambda item: int(item[0]))
        except (TypeError, ValueError):
            return []
        if [int(key) for key, _ in ordered] != list(range(len(ordered))):
            return []
        names = [name for _, name in ordered]
    elif isinstance(value, (list, tuple)):
        names = list(value)
    else:
        return []
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        return []
    return names


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _cancel_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("Cancel marker của inference không hợp lệ.")
    from frameforge.training import _default_app_data_dir

    allowed_parent = (_default_app_data_dir() / "inference" / "cancel").resolve()
    allowed_parent.mkdir(parents=True, exist_ok=True)
    candidate = Path(value).expanduser().resolve()
    if candidate.parent != allowed_parent:
        raise ValueError("Cancel marker nằm ngoài thư mục điều khiển inference của FrameForge.")
    return candidate


def _decode_batch(
    predictions: Any,
    paths: Sequence[Path],
    expected_names: Sequence[str],
    confidence: float,
    selected_classes: Optional[Sequence[int]],
) -> List[Dict[str, Any]]:
    if not isinstance(predictions, (list, tuple)) or len(predictions) != len(paths):
        raise ValueError("Ultralytics trả về số kết quả không khớp số ảnh đầu vào.")
    results = []
    for path, prediction in zip(paths, predictions):
        shape = getattr(prediction, "orig_shape", None)
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("Kết quả inference thiếu kích thước ảnh gốc.")
        height, width = int(shape[0]), int(shape[1])
        if width <= 0 or height <= 0:
            raise ValueError("Kích thước ảnh inference không hợp lệ.")
        boxes = getattr(prediction, "boxes", None)
        items: List[Dict[str, Any]] = []
        skipped_large_objects = 0
        if boxes is not None:
            xyxy = _numpy(getattr(boxes, "xyxy", np.empty((0, 4)))).reshape(-1, 4)
            scores = _numpy(getattr(boxes, "conf", np.empty((0,)))).reshape(-1)
            classes = _numpy(getattr(boxes, "cls", np.empty((0,)))).reshape(-1)
            if not (len(xyxy) == len(scores) == len(classes)):
                raise ValueError("Tensor box/confidence/class từ Ultralytics không đồng bộ.")
            for coordinates, score_value, class_value in zip(xyxy, scores, classes):
                score = float(score_value)
                class_number = float(class_value)
                if not math.isfinite(score) or score < confidence:
                    continue
                if not math.isfinite(class_number) or not class_number.is_integer():
                    raise ValueError("YOLO trả về class ID không hợp lệ.")
                class_id = int(class_number)
                if class_id < 0 or class_id >= len(expected_names):
                    raise ValueError("YOLO trả về class ID ngoài schema của checkpoint.")
                if selected_classes is not None and class_id not in selected_classes:
                    continue
                coordinates = [float(value) for value in coordinates]
                x1, y1, x2, y2 = coordinates
                if any(not math.isfinite(value) for value in coordinates) or not (
                    0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                ):
                    raise ValueError("YOLO trả về bounding box ngoài pixel bounds của ảnh.")
                if (
                    x1 <= CLIPPED_BOX_EDGE_MARGIN_PX
                    or y1 <= CLIPPED_BOX_EDGE_MARGIN_PX
                    or width - x2 <= CLIPPED_BOX_EDGE_MARGIN_PX
                    or height - y2 <= CLIPPED_BOX_EDGE_MARGIN_PX
                ):
                    skipped_large_objects += 1
                    continue
                items.append({
                    "class_id": class_id,
                    "class_name": expected_names[class_id],
                    "confidence": score,
                    "bbox_xyxy": coordinates,
                })
        results.append({
            "image_path": str(path), "width": width, "height": height,
            "predictions": items, "skipped_large_objects": skipped_large_objects,
        })
    return results


def predict_crops(
    checkpoint_value: str,
    image_values: Sequence[str],
    confidence_threshold: Any = 0.25,
    iou_threshold: Any = 0.7,
    class_ids: Any = None,
    cancel_file: Any = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run Ultralytics inference and return pixel-space detections per crop.

    Ultralytics performs its normal NMS once. FrameForge applies the confidence
    and class filters again to guarantee its public filtering semantics.
    """
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("Chọn checkpoint YOLO11n đã đăng ký.")
    try:
        checkpoint = Path(checkpoint_value).expanduser().resolve(strict=True)
    except (OSError, ValueError) as error:
        raise ValueError("Checkpoint không tồn tại.") from error
    if not checkpoint.is_file() or checkpoint.suffix.casefold() != ".pt":
        raise ValueError("Inference chỉ nhận checkpoint `.pt` nội bộ đã xác minh.")
    run = verified_checkpoint_record(checkpoint)
    if run is None:
        raise ValueError("Checkpoint chưa được xác minh trong registry FrameForge hoặc checksum đã thay đổi.")

    expected_names = _names_in_order(run.get("class_names"))
    if not expected_names:
        raise ValueError("Checkpoint thiếu schema class hợp lệ trong registry.")

    if isinstance(confidence_threshold, bool) or not isinstance(confidence_threshold, (int, float)):
        raise ValueError("Confidence threshold phải là số từ 0 đến 1.")
    confidence = float(confidence_threshold)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence threshold phải nằm trong khoảng 0–1.")
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)):
        raise ValueError("NMS IoU phải là số từ 0 đến 1.")
    iou = float(iou_threshold)
    if not math.isfinite(iou) or not 0.0 <= iou <= 1.0:
        raise ValueError("NMS IoU phải nằm trong khoảng 0–1.")

    if not isinstance(image_values, (list, tuple)) or not 1 <= len(image_values) <= MAX_INFERENCE_IMAGES:
        raise ValueError("Inference nhận từ 1 đến {0} ảnh mỗi lượt.".format(MAX_INFERENCE_IMAGES))
    paths: List[Path] = []
    seen = set()
    for value in image_values:
        if not isinstance(value, str) or not value:
            raise ValueError("Đường dẫn ảnh inference không hợp lệ.")
        try:
            path = Path(value).expanduser().resolve(strict=True)
        except (OSError, ValueError) as error:
            raise ValueError("Không tìm thấy ảnh inference: {0}".format(value)) from error
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            raise ValueError("Inference chỉ nhận ảnh được hỗ trợ: {0}".format(path.name))
        if path.stat().st_size <= 0:
            raise ValueError("Ảnh inference rỗng: {0}".format(path.name))
        if path in seen:
            raise ValueError("Danh sách ảnh inference không được trùng đường dẫn.")
        seen.add(path)
        paths.append(path)

    selected_classes: Optional[List[int]] = None
    if class_ids is not None:
        if not isinstance(class_ids, list) or len(class_ids) > len(expected_names):
            raise ValueError("Bộ lọc class inference không hợp lệ.")
        if any(isinstance(class_id, bool) or not isinstance(class_id, int) for class_id in class_ids):
            raise ValueError("Class ID phải là số nguyên.")
        if any(class_id < 0 or class_id >= len(expected_names) for class_id in class_ids):
            raise ValueError("Class ID nằm ngoài schema của checkpoint.")
        if len(set(class_ids)) != len(class_ids):
            raise ValueError("Danh sách class ID không được trùng.")
        selected_classes = class_ids or None

    cancel_path = _cancel_path(cancel_file)
    capabilities = training_capabilities()
    if not capabilities.get("available"):
        raise ValueError("Inference cần PyTorch và Ultralytics trong Python engine.")
    device = capabilities.get("preferred_device", "cpu")
    emit = progress or (lambda _event: None)
    emit({
        "stage": "inference_started",
        "total_images": len(paths),
        "checkpoint_run_id": run["run_id"],
        "device_selected": device,
    })

    try:
        from ultralytics import YOLO

        model = YOLO(str(checkpoint), task="detect")
        actual_names = _names_in_order(getattr(model, "names", None))
        if actual_names != expected_names:
            raise ValueError("Schema class trong checkpoint không khớp registry; dừng inference để tránh sai class ID.")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Không nạp được checkpoint YOLO11n đã xác minh: {0}".format(error)) from error

    results: List[Dict[str, Any]] = []
    cancelled = False
    for start in range(0, len(paths), INFERENCE_BATCH_SIZE):
        if cancel_path and cancel_path.exists():
            cancelled = True
            break
        batch_paths = paths[start:start + INFERENCE_BATCH_SIZE]
        try:
            predictions = model.predict(
                source=[str(path) for path in batch_paths],
                conf=max(0.0, confidence - 1e-9),
                iou=iou,
                classes=selected_classes,
                device=device,
                batch=len(batch_paths),
                max_det=MAX_DETECTIONS_PER_IMAGE,
                verbose=False,
                stream=False,
                save=False,
            )
        except Exception as error:
            raise ValueError("YOLO inference thất bại: {0}".format(error)) from error
        if cancel_path and cancel_path.exists():
            cancelled = True
            break
        results.extend(_decode_batch(predictions, batch_paths, expected_names, confidence, selected_classes))
        emit({
            "stage": "inference_batch_completed",
            "completed_images": len(results),
            "total_images": len(paths),
        })

    emit({
        "stage": "inference_cancelled" if cancelled else "inference_completed",
        "completed_images": len(results),
        "total_images": len(paths),
    })
    return {
        "status": "cancelled" if cancelled else "completed",
        "checkpoint_run_id": run["run_id"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "device_actual": device,
        "processed_images": len(results),
        "total_images": len(paths),
        "skipped_large_objects": sum(item["skipped_large_objects"] for item in results),
        "results": results,
    }


def auto_label_project(
    project_path: str,
    checkpoint_value: str,
    confidence_threshold: Any = 0.25,
    iou_threshold: Any = 0.7,
    cancel_file: Any = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Infer eligible project images once per model load and persist bounded chunks."""
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("Chọn checkpoint YOLO11n đã đăng ký.")
    try:
        checkpoint = Path(checkpoint_value).expanduser().resolve(strict=True)
    except (OSError, ValueError) as error:
        raise ValueError("Checkpoint không tồn tại.") from error
    if not checkpoint.is_file() or checkpoint.suffix.casefold() != ".pt":
        raise ValueError("Inference chỉ nhận checkpoint `.pt` nội bộ đã xác minh.")
    run = verified_checkpoint_record(checkpoint)
    if run is None:
        raise ValueError("Checkpoint chưa được xác minh trong registry FrameForge hoặc checksum đã thay đổi.")
    expected_names = _names_in_order(run.get("class_names"))
    if not expected_names:
        raise ValueError("Checkpoint thiếu schema class hợp lệ trong registry.")
    if isinstance(confidence_threshold, bool) or not isinstance(confidence_threshold, (int, float)):
        raise ValueError("Confidence threshold phải là số từ 0 đến 1.")
    confidence = float(confidence_threshold)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Confidence threshold phải nằm trong khoảng 0–1.")
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)):
        raise ValueError("NMS IoU phải là số từ 0 đến 1.")
    iou = float(iou_threshold)
    if not math.isfinite(iou) or not 0 <= iou <= 1:
        raise ValueError("NMS IoU phải nằm trong khoảng 0–1.")

    project = project_images_for_inference(project_path)
    if project["class_names"] != expected_names:
        raise ValueError("Schema class của checkpoint không khớp với dự án.")
    if len(project["images"]) > MAX_PROJECT_INFERENCE_IMAGES:
        raise ValueError("Dự án vượt giới hạn 100.000 ảnh cho một lượt inference.")
    cancel_path = _cancel_path(cancel_file)
    emit = progress or (lambda _event: None)
    device_info = training_capabilities()
    if not device_info.get("available"):
        raise ValueError("Inference cần PyTorch và Ultralytics trong Python engine.")
    device = device_info.get("preferred_device", "cpu")

    image_rows = []
    skipped_changed = 0
    for item in project["images"]:
        try:
            image_path = Path(item["path"]).expanduser().resolve(strict=True)
            if not image_path.is_file() or image_path.suffix.casefold() not in IMAGE_EXTENSIONS:
                skipped_changed += 1
                continue
            fingerprint = hashlib.sha256(image_path.read_bytes()).hexdigest()
        except OSError:
            skipped_changed += 1
            continue
        image_rows.append({
            "path": image_path,
            "fingerprint": fingerprint,
            "width": int(item["width"]),
            "height": int(item["height"]),
        })

    emit({
        "stage": "project_inference_started",
        "total_images": len(project["images"]),
        "eligible_images": len(image_rows),
        "checkpoint_run_id": run["run_id"],
        "device_selected": device,
    })
    if not image_rows:
        emit({"stage": "project_inference_completed", "completed_images": 0, "total_images": 0})
        return {
            "status": "completed", "checkpoint_run_id": run["run_id"],
            "checkpoint_sha256": run["checkpoint_sha256"], "device_actual": device,
            "total_images": 0, "processed_images": 0, "images_with_detections": 0,
            "candidate_negative_count": 0, "annotation_count": 0, "skipped_large_objects": 0,
            "skipped_changed": skipped_changed, "skipped_duplicate": 0,
        }

    try:
        from ultralytics import YOLO

        model = YOLO(str(checkpoint), task="detect")
        if _names_in_order(getattr(model, "names", None)) != expected_names:
            raise ValueError("Schema class trong checkpoint không khớp registry; dừng inference để tránh sai class ID.")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Không nạp được checkpoint YOLO11n đã xác minh: {0}".format(error)) from error

    checkpoint_stat = checkpoint.stat()
    checkpoint_signature = (checkpoint_stat.st_ino, checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns)
    totals = {
        "processed_images": 0, "images_with_detections": 0,
        "candidate_negative_count": 0, "annotation_count": 0, "skipped_duplicate": 0,
        "skipped_large_objects": 0,
    }
    pending = []
    completed = 0

    def persist_pending() -> None:
        nonlocal skipped_changed
        if not pending:
            return
        try:
            current = checkpoint.stat()
        except OSError as error:
            raise ValueError("Checkpoint biến mất trong lúc inference.") from error
        if (current.st_ino, current.st_size, current.st_mtime_ns) != checkpoint_signature:
            raise ValueError("Checkpoint đã thay đổi trong lúc inference; dừng để tránh sai provenance.")
        saved = _save_ai_suggestions_batch(
            project_path,
            pending,
            str(run["run_id"]),
            str(run["checkpoint_sha256"]),
            expected_names,
        )
        for key in totals:
            totals[key] += saved[key]
        skipped_changed += saved["skipped_changed"]
        pending.clear()
        emit({
            "stage": "project_inference_saved",
            "completed_images": completed,
            "total_images": len(project["images"]),
            "saved_images": totals["processed_images"],
        })

    cancelled = False
    for start in range(0, len(image_rows), INFERENCE_BATCH_SIZE):
        if cancel_path and cancel_path.exists():
            cancelled = True
            break
        batch = image_rows[start:start + INFERENCE_BATCH_SIZE]
        paths = [item["path"] for item in batch]
        try:
            raw_predictions = model.predict(
                source=[str(path) for path in paths],
                conf=max(0.0, confidence - 1e-9),
                iou=iou,
                device=device,
                batch=len(paths),
                max_det=MAX_DETECTIONS_PER_IMAGE,
                verbose=False,
                stream=False,
                save=False,
            )
        except Exception as error:
            raise ValueError("YOLO inference thất bại: {0}".format(error)) from error
        decoded = _decode_batch(raw_predictions, paths, expected_names, confidence, None)
        for row, result in zip(batch, decoded):
            completed += 1
            if result["width"] != row["width"] or result["height"] != row["height"]:
                skipped_changed += 1
                continue
            pending.append({
                "image_path": str(row["path"]),
                "image_fingerprint": row["fingerprint"],
                "predictions": result["predictions"],
                "skipped_large_objects": result["skipped_large_objects"],
            })
            if len(pending) >= PROJECT_PERSIST_BATCH_SIZE:
                persist_pending()
        emit({
            "stage": "project_inference_batch_completed",
            "completed_images": completed,
            "total_images": len(project["images"]),
        })
        if cancel_path and cancel_path.exists():
            cancelled = True
            break

    persist_pending()
    emit({
        "stage": "project_inference_cancelled" if cancelled else "project_inference_completed",
        "completed_images": completed,
        "total_images": len(project["images"]),
    })
    return {
        "status": "cancelled" if cancelled else "completed",
        "checkpoint_run_id": run["run_id"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "device_actual": device,
        "total_images": len(project["images"]),
        **totals,
        "skipped_changed": skipped_changed,
    }
