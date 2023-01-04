import cv2
import numpy as np

class VirtualKeyboard:
    draw_width      : int   = 384
    draw_height     : int   = 256
    last_x          : float = 0.5
    last_y          : float = 0.5
    keyboard_frame  : np.array

    def init(self):
        self.keyboard_frame = np.empty([self.draw_height, self.draw_width], dtype = np.uint8)

    def put_position(self, x : float, y : float):
        print(f"put {x},{y}")
        self.last_x = np.clip(x, 0.0, 1.0)
        self.last_y = np.clip(y, 0.0, 1.0)

    def draw(self):
        keyboard_image = self.keyboard_frame
        keyboard_image.fill(255)

        pos_x = int(self.draw_width  * self.last_x)
        pos_y = int(self.draw_height * self.last_y)

        cv2.circle(keyboard_image, (pos_x, pos_y), 4, 0, 6)
        cv2.imshow("Keyboard", self.keyboard_frame)

