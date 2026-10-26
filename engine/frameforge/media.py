"""Portable media inspection, preview, and frame extraction."""

from __future__ import annotations

import base64
import atexit
import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
FHD_WIDTH = 1920
FHD_HEIGHT = 1080
STRICT_DETAIL_THRESHOLD = .075
STRICT_OBJECT_DETAIL_THRESHOLD = .15


# Thumbnails returned for the gallery; the full image list is always returned separately.
MAX_GALLERY_FILES = 500

def _finite(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) and value > 0 else None


def _require_fhd_video(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    if width > FHD_WIDTH or height > FHD_HEIGHT:
        raise ValueError(
            "Video {0}×{1} vượt giới hạn FHD 1920 × 1080 px; chỉ nhận video tối đa FHD.".format(width, height)
        )


def _read_image(path: Path, flags: int) -> Optional[np.ndarray]:
    """Decode image bytes through Python's Unicode-aware file handling."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


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
        image = _read_image(path, cv2.IMREAD_UNCHANGED)
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


class _VideoPreviewReader:
    """One open video per engine worker; consecutive preview frames avoid reopening and seeking."""

    def __init__(self) -> None:
        self.capture = None
        self.identity = None
        self.next_index = 0

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.identity = None

    def read(self, path: Path, frame_index: int) -> np.ndarray:
        stat = path.stat()
        identity = (str(path), stat.st_size, stat.st_mtime_ns)
        index = max(0, int(frame_index))
        if identity != self.identity:
            self.close()
            self.capture = cv2.VideoCapture(str(path))
            self.identity = identity
            self.next_index = 0
        try:
            if not self.capture.isOpened():
                raise ValueError('Không mở được video nguồn.')
            if index != self.next_index:
                if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise ValueError('Không tua được tới frame đã chọn.')
            ok, frame = self.capture.read()
            if not ok or frame is None:
                raise ValueError('Không đọc được frame {0} trong video.'.format(index))
            self.next_index = index + 1
            _require_fhd_video(frame)
            return frame
        except (ValueError, cv2.error):
            self.close()
            raise


_preview_reader = _VideoPreviewReader()
atexit.register(_preview_reader.close)


def _read_frame(path: Path, frame_index: int) -> np.ndarray:
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        frame = _read_image(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Không đọc được ảnh nguồn.")
        return frame

    return _preview_reader.read(path, frame_index)


MAX_THUMBNAIL_REQUEST = 64


def image_thumbnails(path_values: Any) -> Dict[str, Any]:
    """Return small JPEG thumbnails for project images; unreadable files map to None."""
    if not isinstance(path_values, list) or not 1 <= len(path_values) <= MAX_THUMBNAIL_REQUEST:
        raise ValueError("Mỗi lần chỉ tải từ 1 đến {0} thumbnail.".format(MAX_THUMBNAIL_REQUEST))
    thumbnails: Dict[str, Optional[str]] = {}
    for value in path_values:
        if not isinstance(value, str) or not value:
            raise ValueError("Đường dẫn thumbnail không hợp lệ.")
        path = Path(value).expanduser()
        frame = _read_image(path, cv2.IMREAD_COLOR) if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file() else None
        thumbnails[value] = _thumbnail_data_url(frame) if frame is not None else None
    return {"thumbnails": thumbnails}


def preview_frame(path_value: str, frame_index: int = 0, max_edge: int = 1600, lossless: bool = False, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A frame for display. ``lossless`` (labeling) keeps full resolution and exact pixels as PNG, so
    small objects stay sharp when zoomed; otherwise a JPEG capped at ``max_edge`` (extraction preview)."""
    source_path = Path(path_value).expanduser().resolve(strict=True)
    is_image = source_path.suffix.lower() in IMAGE_EXTENSIONS
    fingerprint_before = hashlib.sha256(source_path.read_bytes()).hexdigest() if is_image else None
    frame = _read_frame(source_path, frame_index)
    image_fingerprint = hashlib.sha256(source_path.read_bytes()).hexdigest() if is_image else None
    if fingerprint_before != image_fingerprint:
        raise ValueError("Ảnh đang được thay đổi; hãy mở preview lại sau khi cập nhật xong.")
    source_height, source_width = frame.shape[:2]
    sharpness = _blur_score(frame)
    analysis = {}
    if options is not None:
        _validate_crop_options(options)
        detector = _cached_preview_detector(options)
        predictions = detector.detect([frame])[0]['predictions'] if detector is not None else []
        crops = _object_crops(frame, options, predictions) if options.get('crop_mode') == 'object_centered' else _crop_frames(frame, options)
        analysis = {
            'predictions': predictions,
            'crop_rects': [item[2] for item in crops],
            'crop_sharpness': [_blur_score(item[3]) for item in crops],
        }
    height, width = frame.shape[:2]
    if lossless:
        ok, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        mime = "image/png"
    else:
        scale = min(1.0, max_edge / max(width, height))
        if scale < 1:
            frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        mime = "image/jpeg"
    if not ok:
        raise ValueError("Không tạo được ảnh preview.")
    data_url = "data:{0};base64,".format(mime) + base64.b64encode(encoded.tobytes()).decode("ascii")
    result = {
        "data_url": data_url,
        "width": width,
        "height": height,
        "source_width": source_width,
        "source_height": source_height,
        "frame_index": max(0, int(frame_index)),
        "sharpness_score": sharpness,
        **analysis,
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


def _detail_score(frame: np.ndarray) -> float:
    """Contrast-normalized edge detail, measured on native pixels rather than a resized preview."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return float(lap.var() / (np.mean(gx * gx + gy * gy) + 1e-6))


def _ssim(left: np.ndarray, right: np.ndarray) -> float:
    a, b = left.astype(np.float32), right.astype(np.float32)
    am, bm = float(a.mean()), float(b.mean())
    covariance = float(np.mean((a-am)*(b-bm)))
    return ((2*am*bm+6.5025)*(2*covariance+58.5225) /
            ((am*am+bm*bm+6.5025)*(float(a.var())+float(b.var())+58.5225)))


class _RecentCrops:
    """Bounded visual confirmation: dHash candidates must also match colors and detected objects."""
    def __init__(self, threshold: int, capacity: int = 4096) -> None:
        self.threshold = threshold
        self.capacity = capacity
        self.masks = _RecentHashes(threshold).masks
        self.buckets: List[Dict[int, set[int]]] = [{} for _ in self.masks]
        self.records: OrderedDict = OrderedDict()
        self.sequence = 0

    @staticmethod
    def signature(frame: np.ndarray, predictions: List[Dict[str, Any]]) -> Tuple:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = []
        for item in sorted(predictions, key=lambda p: (p['class_id'], *p['bbox_xyxy'])):
            x1,y1,x2,y2 = item['bbox_xyxy']
            roi = gray[max(0,int(y1)):min(gray.shape[0],int(math.ceil(y2))), max(0,int(x1)):min(gray.shape[1],int(math.ceil(x2)))]
            if roi.size:
                objects.append((item['class_id'], (x1,y1,x2,y2), cv2.resize(roi,(32,32),interpolation=cv2.INTER_AREA)))
        return cv2.resize(gray,(128,128),interpolation=cv2.INTER_AREA), cv2.resize(frame,(16,16),interpolation=cv2.INTER_AREA), objects

    def contains_near(self, value: int, signature: Tuple) -> bool:
        candidates: set[int] = set(self.records) if self.threshold == 64 else set()
        for (shift,mask), bucket in zip(self.masks,self.buckets):
            candidates.update(bucket.get((value>>shift)&mask,()))
        gray,color,objects = signature
        for identifier in candidates:
            old_hash,(old_gray,old_color,old_objects) = self.records[identifier]
            if (value ^ old_hash).bit_count() > self.threshold or len(objects) != len(old_objects):
                continue
            if float(np.abs(color.astype(np.float32)-old_color).mean()) > 8 or _ssim(gray,old_gray) < .97:
                continue
            # A small changed target can disappear in global SSIM or a 32px thumbnail.
            local_difference = cv2.resize(cv2.absdiff(gray, old_gray), (32, 32), interpolation=cv2.INTER_AREA)
            if int(local_difference.max()) > 12 or int(cv2.absdiff(gray, old_gray).max()) > 32:
                continue
            matching = True
            for (cls,box,roi),(old_cls,old_box,old_roi) in zip(objects,old_objects):
                x1,y1,x2,y2=box; a1,b1,a2,b2=old_box
                overlap=max(0,min(x2,a2)-max(x1,a1))*max(0,min(y2,b2)-max(y1,b1))
                union=(x2-x1)*(y2-y1)+(a2-a1)*(b2-b1)-overlap
                if cls != old_cls or overlap/max(1e-6,union) < .8 or _ssim(roi,old_roi) < .95:
                    matching = False
                    break
            if matching:
                return True
        return False

    def add(self, value: int, signature: Tuple) -> None:
        if len(self.records) >= self.capacity:
            identifier,(old_hash,_) = self.records.popitem(last=False)
            for (shift,mask),bucket in zip(self.masks,self.buckets):
                key=(old_hash>>shift)&mask
                bucket[key].discard(identifier)
                if not bucket[key]: del bucket[key]
        identifier=self.sequence
        self.sequence += 1
        self.records[identifier]=(value,signature)
        for (shift,mask),bucket in zip(self.masks,self.buckets):
            bucket.setdefault((value>>shift)&mask,set()).add(identifier)


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
        raise ValueError("Crop theo vật thể cần checkpoint YOLO11n đã nhập.")
    if mode == "manual":
        raise ValueError("Crop thủ công cần vùng crop đã được người dùng chọn trên preview.")
    raise ValueError("Chế độ crop không hợp lệ.")


def _validate_crop_options(options: Dict[str, Any]) -> None:
    mode = options.get('crop_mode', 'letterbox')
    if mode not in {'center', 'letterbox', 'grid', 'object_centered'}:
        raise ValueError('Chế độ crop chưa khả dụng.')
    minimum = 1 if mode in {'center', 'object_centered'} else 16
    for key, maximum in [('output_width', FHD_WIDTH), ('output_height', FHD_HEIGHT)]:
        size = options.get(key, 416)
        if type(size) is not int or not minimum <= size <= maximum:
            raise ValueError('Kích thước ảnh đầu ra tối đa là 1920 × 1080 px (FHD).')
    if mode == 'object_centered' and not options.get('detection'):
        raise ValueError('Crop theo vật thể cần bật lọc bằng model và chọn checkpoint YOLO11n.')


def _extraction_detector(options: Dict[str, Any]):
    detection = options.get('detection')
    if detection is None:
        return None
    if not isinstance(detection, dict):
        raise ValueError('Cài đặt lọc bằng model không hợp lệ.')
    from frameforge.inference import CropDetector
    return CropDetector(detection.get('checkpoint'), detection.get('confidence_threshold'),
                        detection.get('class_names'), device=detection.get('device', 'auto'))


_preview_detector = None
_preview_detector_key = None


def _cached_preview_detector(options: Dict[str, Any]):
    global _preview_detector, _preview_detector_key
    detection = options.get('detection')
    if detection is None:
        return None
    key = json.dumps(detection, sort_keys=True)
    if _preview_detector is not None and key == _preview_detector_key:
        from frameforge.models import verified_checkpoint_record
        record = verified_checkpoint_record(Path(detection['checkpoint']).expanduser().resolve(strict=True))
        if record is None or record['checkpoint_sha256'] != _preview_detector.checkpoint_sha256:
            _preview_detector = None
            _preview_detector_key = None
            raise ValueError('Checkpoint đã thay đổi hoặc bị xóa khỏi registry; chọn lại model để xem box.')
        return _preview_detector
    _preview_detector = _extraction_detector(options)
    _preview_detector_key = key
    return _preview_detector


def _object_crops(frame: np.ndarray, options: Dict[str, Any], predictions: List[Dict[str, Any]]) -> List[Tuple[np.ndarray, str, Dict[str, int], np.ndarray]]:
    width, height = int(options['output_width']), int(options['output_height'])
    source_height, source_width = frame.shape[:2]
    if width > source_width or height > source_height:
        raise ValueError('Kích thước crop theo vật thể vượt khung nguồn; hãy giảm W/H.')
    crops = []
    seen = set()
    for item in predictions:
        x1, y1, x2, y2 = item['bbox_xyxy']
        if x2 - x1 > width or y2 - y1 > height:
            continue
        left = max(0, min(source_width - width, math.floor((x1 + x2 - width) / 2)))
        top = max(0, min(source_height - height, math.floor((y1 + y2 - height) / 2)))
        if (left, top) in seen:
            continue
        seen.add((left, top))
        crop = frame[top:top + height, left:left + width]
        rect = {'x': left, 'y': top, 'width': width, 'height': height}
        crops.append((crop, 'object_{0:03d}'.format(len(crops) + 1), rect, crop))
    return crops


def _crop_detections(source_result: Dict[str, Any], rect: Dict[str, int]) -> Dict[str, Any]:
    from frameforge.inference import CLIPPED_BOX_EDGE_MARGIN_PX
    predictions = []
    skipped = source_result.get('skipped_large_objects', 0)
    for item in source_result['predictions']:
        x1, y1, x2, y2 = item['bbox_xyxy']
        x1, x2 = x1 - rect['x'], x2 - rect['x']
        y1, y2 = y1 - rect['y'], y2 - rect['y']
        if x2 <= 0 or y2 <= 0 or x1 >= rect['width'] or y1 >= rect['height']:
            continue
        margin = CLIPPED_BOX_EDGE_MARGIN_PX
        if x1 <= margin or y1 <= margin or x2 >= rect['width'] - margin or y2 >= rect['height'] - margin:
            skipped += 1
            continue
        predictions.append({**item, 'bbox_xyxy': [x1, y1, x2, y2]})
    return {'predictions': predictions, 'skipped_large_objects': skipped}


def _source_identity(source: Path) -> str:
    stat = source.stat()
    identity = "{0}\0{1}\0{2}".format(os.path.normcase(str(source.resolve())), stat.st_size, stat.st_mtime_ns)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _sample_capture(capture: cv2.VideoCapture, info: Dict[str, Any], options: Dict[str, Any], statistics: Optional[Dict[str, Any]] = None):
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
        # Decode in order, but only convert selected frames to BGR (costly at FHD).
        if not capture.grab():
            break
        if statistics is not None:
            statistics['scanned_frames'] += 1
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
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                raise ValueError("Không đọc được frame {0} trong video.".format(current_index))
            _require_fhd_video(frame)
            yield current_index, timestamp, frame
            sampled += 1
            if max_frames and sampled >= max_frames:
                break


def _sample_best_capture(capture: cv2.VideoCapture, info: Dict[str, Any], options: Dict[str, Any], statistics: Dict[str, Any]):
    """One native frame per sampling bucket, choosing actual edge detail instead of the first frame."""
    every_frame = {**options, 'interval_unit': 'frames', 'interval': 1, 'max_frames': 0}
    interval = float(options.get('interval',10))
    if not math.isfinite(interval) or interval <= 0: raise ValueError('Khoảng lấy mẫu phải lớn hơn 0.')
    if options.get('interval_unit','frames') not in {'frames','seconds'}: raise ValueError('Đơn vị lấy mẫu không hợp lệ.')
    if options.get('interval_unit','frames') == 'frames' and interval <= 1:
        yield from _sample_capture(capture,info,options,statistics)
        return
    limit = max(0,int(options.get('max_frames',0)))
    group = None
    best = None
    best_score = -1.0
    emitted = 0
    for index,timestamp,frame in _sample_capture(capture,info,every_frame,statistics):
        key = index//max(1,int(interval)) if options.get('interval_unit','frames')=='frames' else int((timestamp-max(0,float(options.get('start_seconds',0))))/interval+1e-6)
        if group is not None and key != group:
            yield best
            emitted += 1
            if limit and emitted >= limit: return
            best, best_score = None, -1.0
        group = key
        if options.get('crop_mode','letterbox') == 'center':
            candidate,_ = _center_crop(frame,int(options.get('output_width',416)),int(options.get('output_height',416)))
        else:
            candidate = frame
        score = _detail_score(candidate)
        if score > best_score:
            best, best_score = (index,timestamp,frame), score
    if best is not None and (not limit or emitted < limit):
        yield best


def _encode_image(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError("Không mã hóa được ảnh đầu ra.")
    return encoded.tobytes()


def _write_bytes(target: Path, data: bytes) -> bool:
    """Write through a temporary file so a partial image never carries the final name."""
    temporary = target.with_name(target.stem + ".partial" + target.suffix)
    try:
        temporary.write_bytes(data)
        os.replace(str(temporary), str(target))
        return True
    except OSError:
        return False
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
    _validate_crop_options(options)
    # Legacy projects may still request JPEG; every new extraction is lossless PNG.
    options = {**options, 'image_format': 'png'}
    options.pop('jpeg_quality', None)
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
        "no_object_crops": 0,
        "detected_images": 0,
        "source_frames_total": 0,
        "scanned_frames": 0,
        "blurry_object_crops": 0,
    }
    exported: List[Dict[str, Any]] = []
    # Every written image is returned without thumbnails so the project registers all of them;
    # only the gallery list is capped.
    written_images: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    duplicate_threshold = max(0, min(64, int(options.get("duplicate_threshold", 1))))
    recent_hashes = _RecentHashes(duplicate_threshold)
    recent_crops = _RecentCrops(duplicate_threshold)
    started = time.monotonic()
    selected_paths, _ = _candidate_paths(source_paths)
    total_sources = len(selected_paths)
    max_frames = max(0, int(options.get("max_frames", 0)))
    extraction_settings = dict(options)
    if options.get('filter_blurry') and options.get('strict_sharpness'):
        extraction_settings['detail_threshold'] = STRICT_DETAIL_THRESHOLD
        extraction_settings['object_detail_threshold'] = STRICT_OBJECT_DETAIL_THRESHOLD
    detection = options.get("detection")
    detector = _extraction_detector(options)
    keep_empty = False
    if detection is not None:
        keep_empty = detection.get("keep_empty") is True
        extraction_settings["detection"] = {
            "checkpoint_sha256": detector.checkpoint_sha256,
            "source_model_id": detector.source_model_id,
            "confidence_threshold": detector.confidence,
            "keep_empty": keep_empty,
            "device_actual": detector.device,
        }

    def emit(source_index: int, source: Path, frame_index: int) -> None:
        if progress is not None:
            progress({
                "source_index": source_index + 1,
                "source_total": total_sources,
                "source_name": source.name,
                "frame_index": frame_index,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                **totals,
                "sampling_skipped_frames": max(0,totals['scanned_frames']-totals['candidate_frames']),
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
        kept = []
        if mode == 'object_centered':
            source_detection = detector.detect([frame])[0]
            crops = _object_crops(frame, options, source_detection['predictions'])
            if not crops:
                totals['no_object_crops'] += 1
        else:
            crops = _crop_frames(frame, options)
        for crop, suffix, crop_rect, filter_crop in crops:
            totals["candidate_crops"] += 1
            sharpness = _blur_score(filter_crop)
            if options.get("filter_blurry") and sharpness < float(options.get("blur_threshold", 60)):
                totals["blurry_crops"] += 1
                continue
            if options.get('filter_blurry') and options.get('strict_sharpness') and _detail_score(filter_crop) < STRICT_DETAIL_THRESHOLD:
                totals['blurry_crops'] += 1
                continue
            current_hash = _dhash(filter_crop)

            base = "{0}_frame_{1:06d}_{2}".format(source.stem, frame_index, suffix)
            target = output / (base + '.png')
            if target.exists() and file_behavior == "skip":
                totals["accepted_crops"] += 1
                accepted_frame = True
                totals["skipped_files"] += 1
                continue
            kept.append((crop, suffix, crop_rect, sharpness, base, target, current_hash))

        # PNG stores these pixels losslessly; inference needs no encode/decode round trip.
        if mode == 'object_centered':
            # PNG crops keep native pixels; translate the source boxes instead of running YOLO again.
            detections = [_crop_detections(source_detection, item[2]) for item in kept]
        else:
            detections = detector.detect([item[0] for item in kept]) if detector is not None else [None] * len(kept)
        for (crop, suffix, crop_rect, sharpness, base, target, current_hash), detected in zip(kept, detections):
            if detected is not None and not detected["predictions"] and not keep_empty:
                # No whole object (boxes touching the crop edge do not count): the crop is not written.
                totals["no_object_crops"] += 1
                continue
            predictions = detected['predictions'] if detected is not None else []
            if options.get('filter_blurry') and options.get('strict_sharpness') and predictions:
                object_scores = []
                for item in predictions:
                    x1,y1,x2,y2 = item['bbox_xyxy']
                    roi = crop[max(0,int(y1)):min(crop.shape[0],int(math.ceil(y2))), max(0,int(x1)):min(crop.shape[1],int(math.ceil(x2)))]
                    if min(roi.shape[:2]) >= 8: object_scores.append(_detail_score(roi))
                if object_scores and min(object_scores) < STRICT_OBJECT_DETAIL_THRESHOLD:
                    totals['blurry_crops'] += 1
                    totals['blurry_object_crops'] += 1
                    continue
            signature = recent_crops.signature(crop,predictions) if options.get('filter_duplicates') and options.get('verify_duplicates', True) else None
            if options.get('filter_duplicates') and (
                recent_crops.contains_near(current_hash,signature) if signature is not None else recent_hashes.contains_near(current_hash)
            ):
                totals['duplicate_crops'] += 1
                continue
            totals["accepted_crops"] += 1
            accepted_frame = True
            if target.exists() and file_behavior == "rename":
                version = 1
                while target.exists():
                    target = output / "{0}_{1}.png".format(base, version)
                    version += 1
            overwritten = target.exists()
            encoded = _encode_image(crop)
            if not _write_bytes(target, encoded):
                raise OSError("Không ghi được ảnh đầu ra: {0}".format(target))
            totals["exported_images"] += 1
            # Only written, quality-checked crops can suppress a later crop as a duplicate.
            if options.get('filter_duplicates'):
                if signature is not None: recent_crops.add(current_hash,signature)
                else: recent_hashes.add(current_hash)
            metadata = {
                "path": str(target),
                "source_path": str(source),
                "source_id": source_id,
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "crop_rect": crop_rect,
                "width": int(crop.shape[1]),
                "height": int(crop.shape[0]),
                "sharpness_score": sharpness,
                "overwritten": overwritten,
            }
            if detected is not None:
                if detected["predictions"]:
                    totals["detected_images"] += 1
                metadata["ai_detection"] = {
                    "image_fingerprint": hashlib.sha256(encoded).hexdigest(),
                    "predictions": [
                        {key: item[key] for key in ("class_id", "confidence", "bbox_xyxy")}
                        for item in detected["predictions"]
                    ],
                    "skipped_large_objects": detected["skipped_large_objects"],
                }
            written_images.append(metadata)
            if len(exported) < MAX_GALLERY_FILES:
                exported.append({
                    **{key: value for key, value in metadata.items() if key != "ai_detection"},
                    "name": target.name,
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
            totals['source_frames_total'] += info.get('frame_count') or 0
            source_id = _source_identity(source)
            if info["kind"] == "image":
                frame = _read_image(source, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Không đọc được ảnh nguồn.")
                totals['scanned_frames'] += 1
                process_frame(source_index, source, source_id, 0, 0.0, frame)
            else:
                capture = cv2.VideoCapture(str(source))
                if not capture.isOpened():
                    raise ValueError("Không mở được video nguồn.")
                try:
                    source_options = dict(options)
                    if max_frames:
                        source_options["max_frames"] = max_frames - totals["candidate_frames"]
                    sampler = _sample_best_capture if options.get('pick_sharpest') else _sample_capture
                    candidates = sampler(capture, info, source_options, totals)
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
        "sampling_skipped_frames": max(0,totals['scanned_frames']-totals['candidate_frames']),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "files": exported,
        "file_list_truncated": totals["exported_images"] > len(exported),
        "images": written_images,
        "extraction_settings": extraction_settings,
        "detection": {
            "source_model_id": detector.source_model_id,
            "checkpoint_sha256": detector.checkpoint_sha256,
            "checkpoint": str(detection.get("checkpoint")),
        } if detector is not None else None,
        "errors": errors,
    }
