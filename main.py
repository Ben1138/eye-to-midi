import cv2

# You can replace 'eye_webcam' with any other
# implementation that implements the Eye class
from eye_webcam import Eye
from virtual_keyboard import VirtualKeyboard

def main():
    eye = Eye()
    keyboard = VirtualKeyboard()

    # Choose appropriate webcam index
    # and number of octaves to cover
    if not eye.init():
        print("Initialization failed!")
        return

    keyboard.init()

    while(True):
        success, x, y = eye.get_screen_position()

        if success:
            keyboard.put_position(x, y)

        keyboard.update_and_draw()
        
        # Exit when pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    eye.shutdown()
    keyboard.shutdown()


if __name__ == '__main__':
    main()