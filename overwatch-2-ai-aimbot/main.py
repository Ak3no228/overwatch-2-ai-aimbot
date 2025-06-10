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
        print(x, y)



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
    # Settings
    AIM_SPEED = 0.42
    CONFIDENCE = 0.45
    COLOR_THRESHOLD = 5
    MAX_DIST_RATIO = 0.31
    MIN_CLOSE_RATIO = 0.12
    HITBOX_POS_RATIO = 0.14
    CYCLES_WAIT_FRAMES = 30

    def __init__(self):
        print("Initializing model...")
        self.model = YOLO("/weights/best.pt")
        print("Model loaded successfully.")

        # Variables
        self.was_target_close = False
        self.cycles_counter = 0
        self.crop_size = 256

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
        
        # Classes
        self.mouse = Mouse()
        self.grabber = ScreenGrabber(self.monitor)


    def run(self):
        print("Aimbot started")
        while True:
            if not self.is_left_mouse_pressed():
                self.was_target_close = False
                self.cycles_counter= 0
                continue

            frame = self.grabber.get_frame()
            if frame is None:
                continue

            target_offset = self.find_target_offset(frame)
            if target_offset:
                self.mouse.move(*target_offset)


    def is_left_mouse_pressed(self):
        return win32api.GetAsyncKeyState(0x01) & 0x8000
    

    def find_target_offset(self, frame):
        results = self.model.predict(
            frame,
            imgsz=self.crop_size,
            conf=self.CONFIDENCE,
            verbose=False
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_box = self.select_best_box(frame, boxes)
        if not best_box:
            return None

        x1, y1, x2, y2 = best_box
        box_center_x = x1 + (x2 - x1) / 2
        box_center_y = y1 + (y2 - y1) * self.HITBOX_POS_RATIO


        # Simple filter for shake between multiple targets, we are waiting 30 frames before switch
        distance = np.sqrt((box_center_x - self.frame_center[0]) ** 2 + (box_center_y - self.frame_center[1]) ** 2)
        target_close = distance < self.max_possible_dist * self.MIN_CLOSE_RATIO

        if target_close:
            self.was_target_close = True
            self.cycles_counter= 0

        elif self.was_target_close:
            if self.cycles_counter< 30:
                self.cycles_counter+= 1
                return None
            else:
                self.was_target_close = False
                self.cycles_counter= 0


        # Convert to offset from the center of the screen and multiply by the aim speed
        offset_x = int((box_center_x - self.frame_center[0]) * self.AIM_SPEED)
        offset_y = int((box_center_y - self.frame_center[1]) * self.AIM_SPEED)

        if offset_x == 0 and offset_y == 0:
            return None
        
        return offset_x, offset_y
    

    def select_best_box(self, frame, boxes):
        best_score = float("inf")
        best_box = None

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # Checking for outline
            if not self.is_valid_target(frame, x1, y1, x2, y2):
                continue

            box_center_x = x1 + (x2 - x1) / 2
            box_center_y = y1 + (y2 - y1) * self.HITBOX_POS_RATIO

            dist_to_center = np.sqrt(
                (box_center_x - self.frame_center[0]) ** 2 +
                (box_center_y - self.frame_center[1]) ** 2
            )

            if dist_to_center > self.max_possible_dist * self.MAX_DIST_RATIO:
                continue

            if dist_to_center < best_score:
                best_score = dist_to_center
                best_box = (x1, y1, x2, y2)

        return best_box
    

    def is_valid_target(self, frame, x1, y1, x2, y2):
        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, self.lower_hsv, self.upper_hsv)
        return cv2.countNonZero(mask) >= self.COLOR_THRESHOLD


if __name__ == "__main__":
    aimbot = Aimbot()
    aimbot.run()
