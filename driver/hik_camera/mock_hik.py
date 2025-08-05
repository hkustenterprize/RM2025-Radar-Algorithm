import cv2
from threading import Lock, Thread
import time


class SimpleHikCamera:
    _instance = None

    def __init__(self, video_source: str):
        """
        Initialize the camera with the given video source.

        Args:
            video_source (int or str): The video source, can be an integer for webcam or a string for video file.
        """
        self.video_source = video_source
        self.capture = cv2.VideoCapture(video_source)
        self.__class__._instance = self

        self.cur_image = None
        self.lock = Lock()
        print(f"Initialized camera with source: {video_source}")
        self.fetch_loop = Thread(target=self.fetch_image_loop, daemon=True)
        self.fetch_loop.start()

    def fetch_image_loop(self):
        while True:
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.cur_image = frame
            else:
                print("Failed to capture image from camera.")
                break
            time.sleep(0.033)

        self.capture.release()

    def get_image_latest(self, group_id: str = "", timeout=1):
        if self.cur_image is None:
            print("No image captured yet.")
            return None, 0.0
        return self.cur_image.copy(), 1.0

    def register_group(self, group_id: str):
        """
        Register a group ID for the camera.

        Args:
            group_id (str): The group ID to register.
        """
        print(f"Registered group ID: {group_id}")
    
    def start_streaming(self):
        pass
    
    def stop_streaming(self):
        pass
