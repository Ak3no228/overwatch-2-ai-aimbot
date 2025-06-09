from ultralytics import YOLO
from threading import Thread
import numpy as np
import win32api
import mss
import cv2


class Mouse:
    def __init__(self):
        pass

    def move(self, x, y):
        """
        Here should be your code for interacting with your driver or Arduino.
        """
        pass


class ScreenGrabber:
    def __init__(self, monitor: dict):
        self.monitor = monitor
        self.current_frame = None
        self.frame_updated = False
        self._start_capture()

    def _start_capture(self):
        thread = Thread(target=self._capture_loop, daemon=True)
        thread.start()

    def _capture_loop(self):
        sct = mss.mss()
        while True:
            screenshot = np.array(sct.grab(self.monitor))
            self.current_frame = screenshot[:, :, :3]
            self.frame_updated = True

    def get_frame(self):
        if self.frame_updated:
            self.frame_updated = False
            return self.current_frame
        return None


class Aimbot:
    def __init__(self):
        # Settings
        self.aim_speed = 0.42
        self.confidence = 0.45
        self.crop_size = 256
        self.color_threshold = 5
        self.max_dist_ratio = 0.31
        self.hitbox_pos_ratio = 0.14

        print("Initializing model...")
        self.model = YOLO("/weights/best.pt")
        print("Model loaded successfully.")

        # Screen center & monitor settings
        self.screen_width = 1920
        self.screen_height = 1080
        self.center_x = self.screen_width // 2
        self.center_y = self.screen_height // 2

        self.monitor = {
            "left": self.center_x - self.crop_size // 2,
            "top": self.center_y - self.crop_size // 2,
            "width": self.crop_size,
            "height": self.crop_size
        }

        # Precomputed values
        self.frame_center = (self.crop_size // 2, self.crop_size // 2)
        self.max_possible_dist = np.sqrt(2 * (self.crop_size ** 2))

        # Target color mask (HSV range)
        self.lower_hsv = np.array([139, 96, 139], np.uint8)
        self.upper_hsv = np.array([157, 255, 255], np.uint8)

        self.mouse = Mouse()
        self.grabber = ScreenGrabber(self.monitor)

    def run(self):
        while True:
            if not self._is_left_mouse_pressed():
                continue

            frame = self.grabber.get_frame()
            if frame is None:
                continue

            target_offset = self._find_target_offset(frame)
            if target_offset:
                self.mouse.move(*target_offset)

    def _is_left_mouse_pressed(self):
        return win32api.GetAsyncKeyState(0x01) & 0x8000

    def _find_target_offset(self, frame):
        results = self.model.predict(
            frame,
            imgsz=self.crop_size,
            conf=self.confidence,
            verbose=False
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_box = self._select_best_box(frame, boxes)
        if not best_box:
            return None

        x1, y1, x2, y2 = best_box
        box_center_x = x1 + (x2 - x1) / 2
        box_center_y = y1 + (y2 - y1) * self.hitbox_pos_ratio

        offset_x = int((box_center_x - self.frame_center[0]) * self.aim_speed)
        offset_y = int((box_center_y - self.frame_center[1]) * self.aim_speed)

        if offset_x == 0 and offset_y == 0:
            return None

        return offset_x, offset_y

    def _select_best_box(self, frame, boxes):
        best_score = float("inf")
        best_box = None

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if not self._is_valid_target(frame, x1, y1, x2, y2):
                continue

            box_center_x = x1 + (x2 - x1) / 2
            box_center_y = y1 + (y2 - y1) * self.hitbox_pos_ratio

            dist_to_center = np.sqrt(
                (box_center_x - self.frame_center[0]) ** 2 +
                (box_center_y - self.frame_center[1]) ** 2
            )

            if dist_to_center > self.max_possible_dist * self.max_dist_ratio:
                continue

            if dist_to_center < best_score:
                best_score = dist_to_center
                best_box = (x1, y1, x2, y2)

        return best_box

    def _is_valid_target(self, frame, x1, y1, x2, y2):
        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, self.lower_hsv, self.upper_hsv)
        return cv2.countNonZero(mask) >= self.color_threshold


if __name__ == "__main__":
    aimbot = Aimbot()
    aimbot.run()