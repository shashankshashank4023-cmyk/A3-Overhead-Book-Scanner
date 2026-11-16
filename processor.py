import cv2
import numpy as np

def order_points(pts):
    pts = np.asarray(pts, dtype="float32")
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")

def four_point_warp(image, pts):
    rect = order_points(pts)
    tl,tr,br,bl = rect
    widthA = np.linalg.norm(br-bl); widthB = np.linalg.norm(tr-tl)
    heightA = np.linalg.norm(tr-br); heightB = np.linalg.norm(tl-bl)
    maxW = max(1, int(max(widthA,widthB))); maxH = max(1, int(max(heightA,heightB)))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect,dst), (maxW,maxH))

def detect_page(image):
    """Find the largest plausible quadrilateral; fall back to original frame."""
    h,w = image.shape[:2]
    ratio = min(1.0, 1400/max(h,w))
    small = cv2.resize(image, (int(w*ratio), int(h*ratio))) if ratio < 1 else image.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=2)
    contours,_ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4 and cv2.contourArea(approx) > small.shape[0]*small.shape[1]*0.08:
            return approx.reshape(4,2).astype("float32") / ratio
    return None

def enhance(image, mode):
    if mode == "Color":
        return cv2.convertScaleAbs(image, alpha=1.05, beta=3)
    if mode == "Grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    adaptive = cv2.adaptiveThreshold(denoised,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 12)
    return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

def process_page(frame, mode="Document"):
    pts = detect_page(frame)
    warped = four_point_warp(frame, pts) if pts is not None else frame.copy()
    return enhance(warped, mode)

def split_spread(image):
    """Split a landscape spread into left and right pages; avoid tiny splits."""
    h,w = image.shape[:2]
    if w < h*1.15: return [image]
    mid = w//2
    return [image[:, :mid].copy(), image[:, mid:].copy()]
