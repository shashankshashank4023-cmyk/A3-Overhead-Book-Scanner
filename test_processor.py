import cv2
import numpy as np
from processor import order_points, four_point_warp, detect_page, enhance, process_page, split_spread


def test_order_points_returns_tl_tr_br_bl():
    pts = np.array([[90, 90], [10, 10], [10, 90], [90, 10]], dtype=np.float32)
    out = order_points(pts)
    assert out.tolist() == [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]]


def test_warp_and_detection_on_synthetic_page():
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.rectangle(image, (120, 70), (680, 530), (245, 245, 245), -1)
    cv2.putText(image, 'A3 TEST PAGE', (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    pts = detect_page(image)
    assert pts is not None and pts.shape == (4, 2)
    warped = four_point_warp(image, pts)
    assert warped.size > 0
    assert warped.shape[0] > 300 and warped.shape[1] > 400


def test_enhancement_modes_keep_valid_image():
    image = np.full((100, 160, 3), 180, dtype=np.uint8)
    for mode in ('Document', 'Color', 'Grayscale'):
        result = enhance(image, mode)
        assert result.shape == image.shape
        assert result.dtype == np.uint8


def test_process_page_fallback_and_spread_split():
    blank = np.zeros((300, 500, 3), dtype=np.uint8)
    assert process_page(blank).shape == blank.shape
    spread = np.zeros((400, 900, 3), dtype=np.uint8)
    pages = split_spread(spread)
    assert len(pages) == 2
    assert pages[0].shape[1] + pages[1].shape[1] == 900
    assert len(split_spread(np.zeros((500, 400, 3), dtype=np.uint8))) == 1
