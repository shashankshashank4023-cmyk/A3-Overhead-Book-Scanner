import cv2

class Camera:
    """UVC camera wrapper. Requests 4K MJPG; gracefully reports actual device mode."""
    def __init__(self, device=0):
        self.device = device
        self.cap = None
        self.running = False
        self.actual = None

    def start(self):
        self.stop()
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            self.cap = None
            raise RuntimeError("Could not open camera. Check USB connection, privacy permissions, and camera index.")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.actual = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.running = True

    def read(self):
        if not self.running or self.cap is None: return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
