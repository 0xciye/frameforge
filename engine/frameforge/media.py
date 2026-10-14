"""Portable media inspection, preview, and frame extraction."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
FHD_WIDTH = 1920
FHD_HEIGHT = 1080


def _finite(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) and value > 0 else None


def _require_fhd_video(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    if width > FHD_WIDTH or height > FHD_HEIGHT:
        raise ValueError(
            "Video {0}×{1} vượt giới hạn FHD 1920 × 1080 px; chỉ nhận video tối đa FHD.".format(width, height)
        )


def _candidate_paths(selected_paths: Iterable[str]) -> Tuple[List[Path], int]:
    paths: List[Path] = []
    unsupported = 0
    seen = set()
    for selected in selected_paths:
        path = Path(selected).expanduser()
        candidates = sorted(path.rglob("*") if path.is_dir() else [path])
        for candidate in candidates:
            suffix = candidate.suffix.lower()
            if not candidate.is_file():
                continue
            if suffix not in MEDIA_EXTENSIONS:
                unsupported += 1
                continue
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                resolved = candidate
            key = os.path.normcase(str(resolved))
            if key in seen:
                continue
            seen.add(key)
            paths.append(resolved)
    return paths, unsupported


def probe_media(path_value: str) -> Dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("Nguồn media không phải là tệp.")
    suffix = path.suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Không đọc được ảnh hoặc định dạng ảnh không hợp lệ.")
        height, width = image.shape[:2]
        return {
            "path": str(path),
            "name": path.name,
            "kind": "image",
            "width": int(width),
            "height": int(height),
            "fps": None,
            "frame_count": 1,
            "duration_seconds": 0,
            "size_bytes": path.stat().st_size,
        }

    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError("Định dạng không được hỗ trợ: {0}".format(suffix or "không có phần mở rộng"))

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Không mở được video. Tệp có thể hỏng hoặc codec chưa được hỗ trợ.")
        fps = _finite(capture.get(cv2.CAP_PROP_FPS))
        frame_count_value = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = int(frame_count_value) if math.isfinite(frame_count_value) and frame_count_value > 0 else None
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise ValueError("Video không có frame đọc được.")
        _require_fhd_video(first_frame)
        height, width = first_frame.shape[:2]
        duration = frame_count / fps if frame_count and fps else None
        if frame_count is None:
            duration_ms = _finite(capture.get(cv2.CAP_PROP_POS_MSEC))
            duration = duration_ms / 1000 if duration_ms else None
        return {
            "path": str(path),
            "name": path.name,
            "kind": "video",
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "size_bytes": path.stat().st_size,
        }
    finally:
        capture.release()


def inspect_sources(selected_paths: Iterable[str]) -> Dict[str, Any]:
    paths, unsupported = _candidate_paths(selected_paths)
    sources = []
    errors = []
    for path in paths:
        try:
            sources.append(probe_media(str(path)))
        except (OSError, ValueError, cv2.error) as error:
            errors.append({"path": str(path), "message": str(error)})
    return {"sources": sources, "unsupported_count": unsupported, "errors": errors}


def _read_frame(path: Path, frame_index: int) -> np.ndarray:
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Không đọc được ảnh nguồn.")
        return frame

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Không mở được video nguồn.")
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError("Không đọc được frame {0} trong video.".format(frame_index))
        _require_fhd_video(frame)
        return frame
    finally:
        capture.release()


def preview_frame(path_value: str, frame_index: int = 0, max_edge: int = 1600) -> Dict[str, Any]:
    source_path = Path(path_value).expanduser().resolve(strict=True)
    is_image = source_path.suffix.lower() in IMAGE_EXTENSIONS
    fingerprint_before = hashlib.sha256(source_path.read_bytes()).hexdigest() if is_image else None
    frame = _read_frame(source_path, frame_index)
    image_fingerprint = hashlib.sha256(source_path.read_bytes()).hexdigest() if is_image else None
    if fingerprint_before != image_fingerprint:
        raise ValueError("Ảnh đang được thay đổi; hãy mở preview lại sau khi cập nhật xong.")
    source_height, source_width = frame.shape[:2]
    height, width = frame.shape[:2]
    scale = min(1.0, max_edge / max(width, height))
    if scale < 1:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError("Không tạo được ảnh preview.")
    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    result = {
        "data_url": data_url,
        "width": width,
        "height": height,
        "source_width": source_width,
        "source_height": source_height,
        "frame_index": max(0, int(frame_index)),
    }
    if image_fingerprint is not None:
        result["image_fingerprint"] = image_fingerprint
    return result


def preview_center_crop(
    path_value: str,
    frame_index: int,
    width: int,
    height: int,
    max_edge: int = 1600,
) -> Dict[str, Any]:
    width = int(width)
    height = int(height)
    if not 1 <= width <= FHD_WIDTH or not 1 <= height <= FHD_HEIGHT:
        raise ValueError("Kích thước crop tối đa là 1920 × 1080 px (FHD).")
    source_path = Path(path_value).expanduser().resolve(strict=True)
    frame = _read_frame(source_path, frame_index)
    source_height, source_width = frame.shape[:2]
    frame_height, frame_width = frame.shape[:2]
    cropped, crop_rect = _center_crop(frame, width, height)
    scale = min(1.0, max_edge / max(width, height))
    if scale < 1:
        cropped = cv2.resize(cropped, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError("Không tạo được preview crop.")
    return {
        "data_url": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
        "width": width,
        "height": height,
        "frame_index": max(0, int(frame_index)),
        "source_width": source_width,
        "source_height": source_height,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "crop_rect": crop_rect,
    }


def inspect_model(path_value: str) -> Dict[str, Any]:
    """Load an ONNX model with the CPU provider and return actual runtime metadata."""
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".onnx":
        raise ValueError("Chọn một tệp model ONNX hợp lệ.")
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("Thiếu ONNX Runtime trong môi trường Python hiện tại.") from error

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map or {}
    class_names = []
    raw_names = metadata.get("names")
    if raw_names:
        try:
            parsed_names = json.loads(raw_names)
            if isinstance(parsed_names, list) and all(isinstance(name, str) for name in parsed_names):
                class_names = parsed_names
            elif isinstance(parsed_names, dict):
                class_names = [str(parsed_names[key]) for key in sorted(parsed_names, key=lambda key: int(key))]
        except (ValueError, TypeError):
            class_names = []

    def tensor(item: Any) -> Dict[str, Any]:
        shape = []
        for dimension in item.shape:
            if isinstance(dimension, int) and dimension > 0:
                shape.append(dimension)
            else:
                shape.append(str(dimension) if dimension else None)
        return {"name": item.name, "shape": shape, "type": item.type}

    return {
        "path": str(path),
        "name": path.name,
        "format": "ONNX",
        "inputs": [tensor(item) for item in session.get_inputs()],
        "outputs": [tensor(item) for item in session.get_outputs()],
        "providers": session.get_providers(),
        "custom_metadata": metadata,
        "class_names": class_names,
    }


def _dhash(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    result = 0
    for bit in bits.ravel():
        result = (result << 1) | int(bit)
    return result


class _RecentHashes:
    """Bounded dHash index for finding nearby accepted frames efficiently."""

    def __init__(self, threshold: int, capacity: int = 4096) -> None:
        self.threshold = max(0, min(64, threshold))
        self.capacity = capacity
        self.values: List[int] = []
        self.buckets: List[Dict[int, set[int]]] = []
        self.masks: List[Tuple[int, int]] = []
        if self.threshold < 64:
            segments = self.threshold + 1
            base, remainder = divmod(64, segments)
            offset = 0
            for index in range(segments):
                width = base + (1 if index < remainder else 0)
                shift = 64 - offset - width
                mask = (1 << width) - 1
                self.masks.append((shift, mask))
                self.buckets.append({})
                offset += width

    def contains_near(self, value: int) -> bool:
        if not self.values:
            return False
        if self.threshold == 64:
            return True
        candidates: set[int] = set()
        for (shift, mask), bucket in zip(self.masks, self.buckets):
            candidates.update(bucket.get((value >> shift) & mask, ()))
        return any((value ^ candidate).bit_count() <= self.threshold for candidate in candidates)

    def add(self, value: int) -> None:
        if value in self.values:
            return
        if len(self.values) >= self.capacity:
            oldest = self.values.pop(0)
            for (shift, mask), bucket in zip(self.masks, self.buckets):
                key = (oldest >> shift) & mask
                members = bucket.get(key)
                if members is not None:
                    members.discard(oldest)
                    if not members:
                        del bucket[key]
        self.values.append(value)
        for (shift, mask), bucket in zip(self.masks, self.buckets):
            bucket.setdefault((value >> shift) & mask, set()).add(value)


def _blur_score(frame: np.ndarray) -> float:
    height, width = frame.shape[:2]
    scale = min(1.0, 640 / max(width, height))
    if scale < 1:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _padding_bgr(options: Dict[str, Any]) -> Tuple[int, int, int]:
    value = options.get("padding_color", "#000000")
    if not isinstance(value, str) or len(value) != 7 or not value.startswith("#"):
        raise ValueError("Màu viền phải có định dạng #RRGGBB.")
    try:
        red, green, blue = (int(value[index : index + 2], 16) for index in (1, 3, 5))
    except ValueError as error:
        raise ValueError("Màu viền phải có định dạng #RRGGBB.") from error
    return blue, green, red


def _letterbox(frame: np.ndarray, width: int, height: int, padding: Tuple[int, int, int]) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    return cv2.copyMakeBorder(
        resized,
        top,
        height - resized_height - top,
        left,
        width - resized_width - left,
        cv2.BORDER_CONSTANT,
        value=padding,
    )


def _axis_starts(source: int, window: int, overlap: float) -> List[int]:
    if source <= window:
        return [0]
    stride = max(1, int(window * (1 - overlap)))
    starts = list(range(0, source - window + 1, stride))
    last = source - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def _center_crop(frame: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, Dict[str, int]]:
    source_height, source_width = frame.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Chiều rộng và chiều cao crop phải lớn hơn 0 px.")
    if width > source_width or height > source_height:
        raise ValueError(
            "Crop {0}×{1} vượt quá khung nguồn {2}×{3}; giảm kích thước crop hoặc chọn nguồn phù hợp.".format(
                width, height, source_width, source_height
            )
        )
    x0 = (source_width - width) // 2
    y0 = (source_height - height) // 2
    x1 = x0 + width
    y1 = y0 + height
    crop = frame[y0:y1, x0:x1]
    if crop.shape[1] != width or crop.shape[0] != height:
        raise ValueError("Frame sau crop không khớp kích thước đã chọn.")
    return crop, {"x": x0, "y": y0, "width": width, "height": height}


def _crop_frames(frame: np.ndarray, options: Dict[str, Any]) -> List[Tuple[np.ndarray, str, Dict[str, int], np.ndarray]]:
    width = int(options.get("output_width", 416))
    height = int(options.get("output_height", 416))
    mode = options.get("crop_mode", "letterbox")
    padding = _padding_bgr(options)
    source_height, source_width = frame.shape[:2]
    full_frame = {"x": 0, "y": 0, "width": source_width, "height": source_height}
    if mode == "center":
        crop, crop_rect = _center_crop(frame, width, height)
        return [(crop, "center", crop_rect, crop)]
    if mode == "letterbox":
        return [(_letterbox(frame, width, height, padding), "full", full_frame, frame)]
    if mode == "grid":
        overlap_x = float(options.get("overlap_x", 0.2))
        overlap_y = float(options.get("overlap_y", 0.2))
        if not 0 <= overlap_x < 1 or not 0 <= overlap_y < 1:
            raise ValueError("Overlap phải nằm trong khoảng từ 0 đến dưới 1.")
        xs = _axis_starts(source_width, width, overlap_x)
        ys = _axis_starts(source_height, height, overlap_y)
        tiles = []
        for row, top in enumerate(ys):
            for column, left in enumerate(xs):
                right = min(source_width, left + width)
                bottom = min(source_height, top + height)
                crop_rect = {"x": left, "y": top, "width": right - left, "height": bottom - top}
                source_crop = frame[top:bottom, left:right]
                tiles.append((
                    _letterbox(source_crop, width, height, padding),
                    "r{0:02d}_c{1:02d}".format(row + 1, column + 1),
                    crop_rect,
                    source_crop,
                ))
        return tiles
    if mode == "object_centered":
        raise ValueError("Crop quanh object cần model ONNX đã nạp; thao tác chưa khả dụng trong phiên này.")
    if mode == "manual":
        raise ValueError("Crop thủ công cần vùng crop đã được người dùng chọn trên preview.")
    raise ValueError("Chế độ crop không hợp lệ.")


def _source_identity(source: Path) -> str:
    stat = source.stat()
    identity = "{0}\0{1}\0{2}".format(os.path.normcase(str(source.resolve())), stat.st_size, stat.st_mtime_ns)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _sample_capture(capture: cv2.VideoCapture, info: Dict[str, Any], options: Dict[str, Any]):
    unit = options.get("interval_unit", "frames")
    interval = float(options.get("interval", 10))
    if interval <= 0:
        raise ValueError("Khoảng lấy mẫu phải lớn hơn 0.")
    start_seconds = max(0.0, float(options.get("start_seconds", 0)))
    end_seconds = options.get("end_seconds")
    end_seconds = float(end_seconds) if end_seconds is not None else None
    max_frames = max(0, int(options.get("max_frames", 0)))
    fps = info.get("fps")
    next_timestamp = start_seconds
    frame_index = 0
    sampled = 0
    last_timestamp = -1.0

    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        _require_fhd_video(frame)
        reported = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        fallback = frame_index / fps if fps else float(frame_index)
        timestamp = reported if math.isfinite(reported) and reported > last_timestamp else fallback
        last_timestamp = timestamp
        current_index = frame_index
        frame_index += 1

        if timestamp < start_seconds:
            continue
        if end_seconds is not None and timestamp > end_seconds:
            break
        if unit == "frames":
            frame_step = max(1, int(interval))
            selected = current_index % frame_step == 0
        elif unit == "seconds":
            selected = timestamp + 1e-6 >= next_timestamp
            if selected:
                next_timestamp = timestamp + interval
        else:
            raise ValueError("Đơn vị lấy mẫu không hợp lệ.")
        if selected:
            yield current_index, timestamp, frame
            sampled += 1
            if max_frames and sampled >= max_frames:
                break


def _write_image(frame: np.ndarray, destination: Path, image_format: str, quality: int) -> bool:
    if image_format == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        extension = ".png"
    else:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        extension = ".jpg"
    target = destination.with_suffix(extension)
    temporary = target.with_name(target.stem + ".partial" + target.suffix)
    try:
        if not cv2.imwrite(str(temporary), frame, params):
            return False
        os.replace(str(temporary), str(target))
        return True
    finally:
        if temporary.exists():
            temporary.unlink()


def _thumbnail_data_url(frame: np.ndarray, max_edge: int = 180) -> str:
    height, width = frame.shape[:2]
    scale = min(1.0, max_edge / max(width, height))
    thumbnail = frame if scale == 1 else cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    ok, encoded = cv2.imencode(".jpg", thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 68])
    if not ok:
        raise ValueError("Không tạo được thumbnail cho ảnh vừa trích.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def extract_frames(
    source_paths: Iterable[str],
    output_dir: str,
    options: Dict[str, Any],
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    width = int(options.get("output_width", 416))
    height = int(options.get("output_height", 416))
    mode = options.get("crop_mode", "letterbox")
    if mode not in {"center", "letterbox", "grid"}:
        raise ValueError("Chế độ crop chưa khả dụng.")
    minimum_size = 1 if mode == "center" else 16
    if not minimum_size <= width <= FHD_WIDTH or not minimum_size <= height <= FHD_HEIGHT:
        raise ValueError("Kích thước ảnh đầu ra tối đa là 1920 × 1080 px (FHD).")
    image_format = options.get("image_format", "jpg")
    if image_format not in {"jpg", "png"}:
        raise ValueError("Định dạng đầu ra chỉ hỗ trợ JPG hoặc PNG.")
    quality = int(options.get("jpeg_quality", 92))
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality phải nằm trong khoảng 1–100.")
    file_behavior = options.get("existing_file_behavior", "skip")
    if file_behavior not in {"skip", "overwrite", "rename"}:
        raise ValueError("Chính sách xử lý tệp trùng không hợp lệ.")

    totals = {
        "candidate_frames": 0,
        "accepted_frames": 0,
        "candidate_crops": 0,
        "accepted_crops": 0,
        "duplicate_crops": 0,
        "blurry_crops": 0,
        "exported_images": 0,
        "skipped_files": 0,
        "failed_sources": 0,
    }
    exported: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    duplicate_threshold = max(0, min(64, int(options.get("duplicate_threshold", 1))))
    recent_hashes = _RecentHashes(duplicate_threshold)
    started = time.monotonic()
    selected_paths, _ = _candidate_paths(source_paths)
    total_sources = len(selected_paths)
    max_frames = max(0, int(options.get("max_frames", 0)))
    extraction_settings = dict(options)

    def emit(source_index: int, source: Path, frame_index: int) -> None:
        if progress is not None:
            progress({
                "source_index": source_index + 1,
                "source_total": total_sources,
                "source_name": source.name,
                "frame_index": frame_index,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                **totals,
            })

    def process_frame(
        source_index: int,
        source: Path,
        source_id: str,
        frame_index: int,
        timestamp: float,
        frame: np.ndarray,
    ) -> None:
        totals["candidate_frames"] += 1
        accepted_frame = False
        for crop, suffix, crop_rect, filter_crop in _crop_frames(frame, options):
            totals["candidate_crops"] += 1
            sharpness = _blur_score(filter_crop)
            if options.get("filter_blurry") and sharpness < float(options.get("blur_threshold", 60)):
                totals["blurry_crops"] += 1
                continue
            current_hash = _dhash(filter_crop)
            if options.get("filter_duplicates") and recent_hashes.contains_near(current_hash):
                totals["duplicate_crops"] += 1
                continue
            if options.get("filter_duplicates"):
                recent_hashes.add(current_hash)

            totals["accepted_crops"] += 1
            accepted_frame = True
            base = "{0}_frame_{1:06d}_{2}".format(source.stem, frame_index, suffix)
            target = (output / base).with_suffix(".png" if image_format == "png" else ".jpg")
            if target.exists() and file_behavior == "skip":
                totals["skipped_files"] += 1
                continue
            if target.exists() and file_behavior == "rename":
                version = 1
                while target.exists():
                    target = output / "{0}_{1}{2}".format(base, version, ".png" if image_format == "png" else ".jpg")
                    version += 1
            if not _write_image(crop, target, image_format, quality):
                raise OSError("Không ghi được ảnh đầu ra: {0}".format(target))
            totals["exported_images"] += 1
            if len(exported) < 500:
                exported.append({
                    "path": str(target),
                    "source_path": str(source),
                    "source_id": source_id,
                    "name": target.name,
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "crop_rect": crop_rect,
                    "width": int(crop.shape[1]),
                    "height": int(crop.shape[0]),
                    "sharpness_score": sharpness,
                    "extraction_settings": extraction_settings,
                    "thumbnail_data_url": _thumbnail_data_url(crop),
                })
        if accepted_frame:
            totals["accepted_frames"] += 1
        if totals["candidate_frames"] % 20 == 0:
            emit(source_index, source, frame_index)

    for source_index, source in enumerate(selected_paths):
        if max_frames and totals["candidate_frames"] >= max_frames:
            break
        try:
            info = probe_media(str(source))
            source_id = _source_identity(source)
            if info["kind"] == "image":
                frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Không đọc được ảnh nguồn.")
                process_frame(source_index, source, source_id, 0, 0.0, frame)
            else:
                capture = cv2.VideoCapture(str(source))
                if not capture.isOpened():
                    raise ValueError("Không mở được video nguồn.")
                try:
                    source_options = dict(options)
                    if max_frames:
                        source_options["max_frames"] = max_frames - totals["candidate_frames"]
                    candidates = _sample_capture(capture, info, source_options)
                    for frame_index, timestamp, frame in candidates:
                        process_frame(source_index, source, source_id, frame_index, timestamp, frame)
                finally:
                    capture.release()
                emit(source_index, source, -1)
        except (OSError, ValueError, cv2.error) as error:
            totals["failed_sources"] += 1
            errors.append({"path": str(source), "message": str(error)})
        emit(source_index, source, -1)

    return {
        **totals,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "files": exported,
        "file_list_truncated": totals["exported_images"] > len(exported),
        "errors": errors,
    }
