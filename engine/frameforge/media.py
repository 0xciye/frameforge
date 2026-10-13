"""Portable media inspection, preview, and frame extraction."""

from __future__ import annotations

import base64
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


def _finite(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) and value > 0 else None


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
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if frame_count and fps else None
        if frame_count is None:
            duration_ms = _finite(capture.get(cv2.CAP_PROP_POS_MSEC))
            duration = duration_ms / 1000 if duration_ms else None
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise ValueError("Video không có frame đọc được.")
        return {
            "path": str(path),
            "name": path.name,
            "kind": "video",
            "width": width or int(first_frame.shape[1]),
            "height": height or int(first_frame.shape[0]),
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
        return frame
    finally:
        capture.release()


def preview_frame(path_value: str, frame_index: int = 0, max_edge: int = 1600) -> Dict[str, Any]:
    frame = _read_frame(Path(path_value).expanduser().resolve(strict=True), frame_index)
    source_height, source_width = frame.shape[:2]
    scale = min(1.0, max_edge / max(source_width, source_height))
    if scale < 1:
        frame = cv2.resize(frame, (round(source_width * scale), round(source_height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError("Không tạo được ảnh preview.")
    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    return {
        "data_url": data_url,
        "width": source_width,
        "height": source_height,
        "frame_index": max(0, int(frame_index)),
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


def _crop_frames(frame: np.ndarray, options: Dict[str, Any]) -> List[Tuple[np.ndarray, str]]:
    width = int(options.get("output_width", 416))
    height = int(options.get("output_height", 416))
    mode = options.get("crop_mode", "letterbox")
    padding = _padding_bgr(options)
    if mode == "letterbox":
        return [(_letterbox(frame, width, height, padding), "full")]
    if mode == "grid":
        source_height, source_width = frame.shape[:2]
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
                tiles.append((_letterbox(frame[top:bottom, left:right], width, height, padding), "r{0:02d}_c{1:02d}".format(row + 1, column + 1)))
        return tiles
    if mode == "object_centered":
        raise ValueError("Crop quanh object cần model ONNX đã nạp; thao tác chưa khả dụng trong phiên này.")
    if mode == "manual":
        raise ValueError("Crop thủ công cần vùng crop đã được người dùng chọn trên preview.")
    raise ValueError("Chế độ crop không hợp lệ.")


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
    if not 16 <= width <= 4096 or not 16 <= height <= 4096:
        raise ValueError("Kích thước đầu ra phải nằm trong khoảng 16–4096 px.")
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
        "duplicate_frames": 0,
        "blurry_frames": 0,
        "exported_images": 0,
        "skipped_files": 0,
        "failed_sources": 0,
    }
    exported: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    duplicate_threshold = max(0, min(64, int(options.get("duplicate_threshold", 5))))
    recent_hashes = _RecentHashes(duplicate_threshold)
    started = time.monotonic()
    selected_paths, _ = _candidate_paths(source_paths)
    total_sources = len(selected_paths)
    max_frames = max(0, int(options.get("max_frames", 0)))

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

    for source_index, source in enumerate(selected_paths):
        if max_frames and totals["candidate_frames"] >= max_frames:
            break
        try:
            info = probe_media(str(source))
            if info["kind"] == "image":
                frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
                candidates = [(0, 0.0, frame)] if frame is not None else []
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
                        totals["candidate_frames"] += 1
                        if options.get("filter_blurry") and _blur_score(frame) < float(options.get("blur_threshold", 60)):
                            totals["blurry_frames"] += 1
                            if totals["candidate_frames"] % 20 == 0:
                                emit(source_index, source, frame_index)
                            continue
                        current_hash = _dhash(frame)
                        if options.get("filter_duplicates") and recent_hashes.contains_near(current_hash):
                            totals["duplicate_frames"] += 1
                            if totals["candidate_frames"] % 20 == 0:
                                emit(source_index, source, frame_index)
                            continue
                        if options.get("filter_duplicates"):
                            recent_hashes.add(current_hash)
                        totals["accepted_frames"] += 1
                        tiles = _crop_frames(frame, options)
                        for tile, suffix in tiles:
                            base = "{0}_frame_{1:06d}_{2}".format(source.stem, frame_index, suffix)
                            target = output / base
                            extension = ".png" if image_format == "png" else ".jpg"
                            target = target.with_suffix(extension)
                            if target.exists() and file_behavior == "skip":
                                totals["skipped_files"] += 1
                                continue
                            if target.exists() and file_behavior == "rename":
                                version = 1
                                while target.exists():
                                    target = output / "{0}_{1}{2}".format(base, version, extension)
                                    version += 1
                            if not _write_image(tile, target, image_format, quality):
                                raise OSError("Không ghi được ảnh đầu ra: {0}".format(target))
                            totals["exported_images"] += 1
                            if len(exported) < 500:
                                exported.append({
                                    "path": str(target),
                                    "source_path": str(source),
                                    "name": target.name,
                                    "frame_index": frame_index,
                                    "timestamp_seconds": timestamp,
                                    "width": width,
                                    "height": height,
                                    "thumbnail_data_url": _thumbnail_data_url(tile),
                                })
                        if totals["candidate_frames"] % 20 == 0:
                            emit(source_index, source, frame_index)
                finally:
                    capture.release()
                emit(source_index, source, -1)
                continue

            if not candidates or candidates[0][2] is None:
                raise ValueError("Không đọc được ảnh nguồn.")
            frame_index, timestamp, frame = candidates[0]
            totals["candidate_frames"] += 1
            if options.get("filter_blurry") and _blur_score(frame) < float(options.get("blur_threshold", 60)):
                totals["blurry_frames"] += 1
                continue
            current_hash = _dhash(frame)
            if options.get("filter_duplicates") and recent_hashes.contains_near(current_hash):
                totals["duplicate_frames"] += 1
                continue
            if options.get("filter_duplicates"):
                recent_hashes.add(current_hash)
            totals["accepted_frames"] += 1
            for tile, suffix in _crop_frames(frame, options):
                base = "{0}_{1}".format(source.stem, suffix)
                target = output / base
                extension = ".png" if image_format == "png" else ".jpg"
                target = target.with_suffix(extension)
                if target.exists() and file_behavior == "skip":
                    totals["skipped_files"] += 1
                    continue
                if target.exists() and file_behavior == "rename":
                    version = 1
                    while target.exists():
                        target = output / "{0}_{1}{2}".format(base, version, extension)
                        version += 1
                if not _write_image(tile, target, image_format, quality):
                    raise OSError("Không ghi được ảnh đầu ra: {0}".format(target))
                totals["exported_images"] += 1
                if len(exported) < 500:
                    exported.append({
                        "path": str(target),
                        "source_path": str(source),
                        "name": target.name,
                        "frame_index": 0,
                        "timestamp_seconds": 0.0,
                        "width": width,
                        "height": height,
                        "thumbnail_data_url": _thumbnail_data_url(tile),
                    })
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
