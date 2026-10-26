import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frameforge import media
from test_engine import create_video,distinct_frame


def test_legacy_jpeg_request_writes_exact_png_pixels(tmp_path):
    image=np.random.default_rng(3).integers(0,256,(48,64,3),dtype=np.uint8)
    source=tmp_path/'source.png'; cv2.imencode('.png',image)[1].tofile(source)
    result=media.extract_frames([str(source)],str(tmp_path/'out'),dict(crop_mode='center',output_width=32,
        output_height=24,image_format='jpg',jpeg_quality=1))
    assert not result['errors'] and result['exported_images']==1
    path=Path(result['images'][0]['path'])
    assert path.suffix=='.png' and path.read_bytes().startswith(b'\x89PNG')
    np.testing.assert_array_equal(cv2.imread(str(path)),image[12:36,16:48])
    assert result['extraction_settings']['image_format']=='png'
    assert 'jpeg_quality' not in result['extraction_settings']


def test_detail_score_detects_blur_despite_a_high_laplacian():
    sharp=np.random.default_rng(5).integers(0,256,(128,128,3),dtype=np.uint8)
    soft=cv2.GaussianBlur(sharp,(7,7),1.4)
    assert media._detail_score(sharp)>media.STRICT_DETAIL_THRESHOLD
    assert media._detail_score(soft)<media.STRICT_DETAIL_THRESHOLD


@pytest.mark.parametrize('unit,interval',[('frames',2),('seconds',.2)])
def test_best_sampling_keeps_native_pixels_and_counts_skips(tmp_path,unit,interval):
    source=tmp_path/'sample.avi'; create_video(source,[distinct_frame(i) for i in range(8)])
    capture=cv2.VideoCapture(str(source)); decoded=[]
    while True:
        ok,frame=capture.read()
        if not ok: break
        decoded.append(frame)
    capture.release()
    capture=cv2.VideoCapture(str(source)); stats={'scanned_frames':0}
    try:
        sampled=list(media._sample_best_capture(capture,{'fps':10},dict(interval_unit=unit,interval=interval,
            crop_mode='center',output_width=32,output_height=24),stats))
    finally: capture.release()
    expected=[max(range(i,i+2),key=lambda n:media._detail_score(decoded[n][12:36,16:48])) for i in range(0,8,2)]
    assert [i for i,_,_ in sampled]==expected
    assert stats['scanned_frames']==8
    for i,_,frame in sampled: np.testing.assert_array_equal(frame,decoded[i])
    result=media.extract_frames([str(source)],str(tmp_path/'out'),dict(interval_unit=unit,interval=interval,
        crop_mode='center',output_width=32,output_height=24,pick_sharpest=True))
    assert result['candidate_frames']==4 and result['source_frames_total']==8
    assert result['sampling_skipped_frames']==4 and result['scanned_frames']==8


def test_visual_duplicates_keep_changed_tiny_targets():
    old=np.full((640,640,3),80,dtype=np.uint8); old[40:44,40:44]=240
    new=np.full((640,640,3),80,dtype=np.uint8); new[100:104,100:104]=240
    a=media._RecentCrops.signature(old,[dict(class_id=0,bbox_xyxy=[40,40,44,44])])
    b=media._RecentCrops.signature(new,[dict(class_id=0,bbox_xyxy=[100,100,104,104])])
    assert media._ssim(a[0],b[0])>.97
    cache=media._RecentCrops(6,capacity=1); cache.add(0,a)
    assert cache.contains_near(0,a)
    assert not cache.contains_near(0,b)
    cache.add(0,b)
    assert len(cache.records)==1 and not cache.contains_near(0,a)
    assert not cache.contains_near(127,b)  # seven changed bits exceed the chosen six.


def test_visual_duplicates_do_not_merge_different_colors():
    a=media._RecentCrops.signature(np.full((32,32,3),(240,0,0),dtype=np.uint8),[])
    b=media._RecentCrops.signature(np.full((32,32,3),(0,0,90),dtype=np.uint8),[])
    cache=media._RecentCrops(64); cache.add(0,a)
    assert not cache.contains_near(0,b)
