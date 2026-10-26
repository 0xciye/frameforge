import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frameforge import media
from test_engine import create_video, distinct_frame
from test_inference import register_checkpoint, install_brightness_yolo, extraction_options, extraction_sources


def test_object_crop_detects_only_source_and_translates_boxes(tmp_path, monkeypatch):
    monkeypatch.setenv('FRAMEFORGE_APP_DATA', str(tmp_path / 'app-data'))
    checkpoint = register_checkpoint(tmp_path)
    model = install_brightness_yolo(monkeypatch)
    sources = extraction_sources(tmp_path)
    options = {**extraction_options(checkpoint), 'crop_mode': 'object_centered'}
    result = media.extract_frames(sources, str(tmp_path / 'out'), options)
    assert not result['errors']
    assert result['exported_images'] == 1 and result['no_object_crops'] == 1
    calls = model.instances[0].calls
    assert len(model.instances) == 1 and len(calls) == 2
    assert all(len(call['source']) == 1 and call['source'][0].shape == (48, 64, 3) for call in calls)
    image = result['images'][0]
    rect = image['crop_rect']
    np.testing.assert_array_equal(cv2.imread(image['path']), cv2.imread(sources[0])[rect['y']:rect['y'] + 24, rect['x']:rect['x'] + 32])
    assert image['ai_detection']['predictions'][0]['bbox_xyxy'] == [4, 4, 20, 18]

    preview = media.preview_frame(sources[0], max_edge=16, options=options)
    assert preview['crop_rects'] == [rect]
    assert preview['predictions'][0]['bbox_xyxy'] == [4, 4, 20, 18]
    assert preview['sharpness_score'] == media._blur_score(cv2.imread(sources[0]))


def test_object_crop_clamps_edges_skips_oversized_and_merges_identical_windows():
    frame = np.zeros((80, 120, 3), np.uint8)
    options = {'output_width': 40, 'output_height': 30}
    predictions = [{'bbox_xyxy': box} for box in ([90, 60, 118, 78], [90, 60, 118, 78], [1, 1, 60, 40])]
    crops = media._object_crops(frame, options, predictions)
    assert len(crops) == 1
    assert crops[0][2] == {'x': 80, 'y': 50, 'width': 40, 'height': 30}
    result = media._crop_detections({'predictions': [
        {'class_id': 0, 'confidence': .9, 'bbox_xyxy': [90, 60, 118, 78]},
        {'class_id': 1, 'confidence': .8, 'bbox_xyxy': [79, 55, 100, 70]},
        {'class_id': 0, 'confidence': .7, 'bbox_xyxy': [1, 1, 10, 10]},
    ]}, crops[0][2])
    assert result['predictions'][0]['bbox_xyxy'] == [10, 10, 38, 28]
    assert len(result['predictions']) == 1 and result['skipped_large_objects'] == 1


def test_object_crop_requires_model_before_writing(tmp_path):
    with pytest.raises(ValueError, match='cần bật lọc bằng model'):
        media.extract_frames([], str(tmp_path / 'out'), {'crop_mode': 'object_centered'})
    assert not list((tmp_path / 'out').glob('*.png'))


def test_duplicate_object_crops_are_rejected_before_png_encoding(tmp_path, monkeypatch):
    monkeypatch.setenv('FRAMEFORGE_APP_DATA', str(tmp_path / 'app-data'))
    checkpoint = register_checkpoint(tmp_path)
    model = install_brightness_yolo(monkeypatch)
    sources = extraction_sources(tmp_path)
    duplicate = tmp_path / 'same-frame.png'
    duplicate.write_bytes(Path(sources[0]).read_bytes())
    encode = media._encode_image
    encoded = []

    def track(frame):
        encoded.append(frame.shape)
        return encode(frame)

    monkeypatch.setattr(media, '_encode_image', track)
    options = {**extraction_options(checkpoint), 'crop_mode': 'object_centered', 'filter_duplicates': True}
    result = media.extract_frames([sources[0], str(duplicate), sources[1]], str(tmp_path / 'out'), options)
    assert not result['errors']
    assert result['exported_images'] == 1 and result['duplicate_crops'] == 1 and result['no_object_crops'] == 1
    assert encoded == [(24, 32, 3)]
    assert len(model.instances[0].calls) == 3


def test_duplicate_verification_preserves_small_target_without_model():
    old = np.full((640, 640, 3), 80, np.uint8)
    old[40:44, 40:44] = 240
    changed = np.full_like(old, 80)
    changed[100:104, 100:104] = 240
    cache = media._RecentCrops(6)
    cache.add(0, cache.signature(old, []))
    assert cache.contains_near(0, cache.signature(old.copy(), []))
    assert not cache.contains_near(0, cache.signature(changed, []))


def test_preview_reuses_model_but_rejects_modified_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('FRAMEFORGE_APP_DATA', str(tmp_path / 'app-data'))
    monkeypatch.setattr(media, '_preview_detector', None)
    monkeypatch.setattr(media, '_preview_detector_key', None)
    checkpoint = register_checkpoint(tmp_path)
    model = install_brightness_yolo(monkeypatch)
    source = extraction_sources(tmp_path)[0]
    options = {**extraction_options(checkpoint), 'crop_mode': 'object_centered'}
    for _ in range(3):
        result = media.preview_frame(source, options=options)
        assert result['predictions'][0]['bbox_xyxy'] == [4, 4, 20, 18]
    assert len(model.instances) == 1 and len(model.instances[0].calls) == 3
    options['detection']['confidence_threshold'] = .5
    media.preview_frame(source, options=options)
    assert len(model.instances) == 2
    checkpoint.write_bytes(b'changed-checkpoint')
    with pytest.raises(ValueError, match='Checkpoint đã thay đổi'):
        media.preview_frame(source, options=options)


def test_video_preview_reuses_decoder_and_seeks_only_when_needed(tmp_path, monkeypatch):
    path = tmp_path / 'preview.avi'
    create_video(path, [distinct_frame(i) for i in range(6)])
    original = cv2.VideoCapture
    reference = original(str(path))
    expected = []
    while True:
        ok, frame = reference.read()
        if not ok:
            break
        expected.append(frame)
    reference.release()
    opened, seeks = [], []

    class TrackedCapture:
        def __init__(self, path):
            self.capture = original(path)
            opened.append(path)

        def isOpened(self): return self.capture.isOpened()
        def read(self): return self.capture.read()
        def release(self): self.capture.release()
        def set(self, prop, index):
            seeks.append(index)
            return self.capture.set(prop, index)

    monkeypatch.setattr(cv2, 'VideoCapture', TrackedCapture)
    reader = media._VideoPreviewReader()
    try:
        for index in [0, 1, 2, 4, 0]:
            np.testing.assert_array_equal(reader.read(path, index), expected[index])
        assert len(opened) == 1 and seeks == [4, 0]
        with pytest.raises(ValueError, match='Không đọc được frame'):
            reader.read(path, 100)
        assert reader.capture is None
    finally:
        reader.close()
