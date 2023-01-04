import cv2
import numpy as np
import dlib

# Implementation for webcam
class Eye:
    video = None

    # Does not take any arguments
    def init(self) -> bool:
        assert self.video    == None

        try:
            # For me, my webcam was at index 1, but it's usually 0
            self.video = cv2.VideoCapture(0)
        except:
            print("video initialization failed!")
            return False

        assert self.video != None

        gpu_count = dlib.cuda.get_num_devices()
        print("Num available GPUs:", gpu_count)
        if gpu_count < 1:
            print("No GPU available for CUDA!")
            return False

        if not dlib.DLIB_USE_CUDA:
            print("dlib is not using CUDA!")
            return False
        
        def on_changed(value):
            pass

        cv2.namedWindow('Frame')
        cv2.createTrackbar("Narrowing",   "Frame",   0,  10, on_changed)
        cv2.createTrackbar("Threshold",   "Frame",   0, 255, on_changed)
        cv2.createTrackbar("Blur",        "Frame",   0,  10, on_changed)
        cv2.createTrackbar("Width Scale", "Frame", 100, 300, on_changed)
        cv2.setTrackbarMin("Width Scale", "Frame", 100)
        return self.__calibrate()


    # Does not take any arguments
    def shutdown(self):
        assert self.video != None

        self.video.release()
        cv2.destroyAllWindows()


    # Returns the position on a virtual screen of where the participant is looking  
    # at in normalized 0-1 2D space, with the origin being at the top left
    #
    #   0,0        1,0
    #    ┌──────────┐
    #    │          │
    #    │          │
    #    └──────────┘
    #   0,1        1,1
    #
    def get_screen_position(self) -> tuple[bool, float, float]:
        assert self.video != None

        success, frame_bgr, frame_gray = self.__get_frame()
        if not success:
            print("Reading frame from video source failed!")
            return False, 0, 0

        success, pupils = self.__get_pupil_positions(frame_bgr, frame_gray)
        cv2.imshow("Frame", frame_bgr)

        if success:
            assert len(pupils) == 2
            left_x  = pupils[0]
            right_x = pupils[1]

            gaze_x = (left_x + right_x) / 2.0
            gaze_y = 0.5

            return True, gaze_x, gaze_y

        return False, 0.5, 0.5


    # Private method
    def __get_frame(self) -> tuple[bool, np.ndarray, np.ndarray]:
        assert self.video != None

        success, frame_bgr = self.video.read()
        if not success:
            return False, None, None

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        # frame_gray = cv2.resize(frame_gray, None, fx=0.5, fy=0.5)
        # frame_bgr  = cv2.resize(frame_bgr,  None, fx=0.5, fy=0.5)

        return True, frame_bgr, frame_gray


    # Private method
    def __calibrate(self) -> bool: 
        assert self.video != None

        # while(True):
        #     success, frame_gray = self.__get_gray_frame()
        #     if not success:
        #         print("Reading frame from video source failed!")
        #         return False

        #     success, pupils = self.__get_pupil_positions(frame_gray)

        #     cv2.imshow("Calibration", frame_gray)
            
        #     # Exit when pressing 'q'
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        
        # cv2.destroyAllWindows()
        return True


    # Private method
    # Returns True and the two pupil x-positions within local eye-space 
    # or False when no face or no pupils have been found
    # 
    #    0.0           1.0          looking right         looking left
    #     ┌──────┬──────┐          ┌─────────┬──┐        ┌─┬──────────┐
    #     │      │      │          │         │  │        │ │          │
    #     ├──────O──────┤          ├─────────O──┤        ├─O──────────┤
    #     │      │      │          │         │  │        │ │          │
    #     └──────┴──────┘          └─────────┴──┘        └─┴──────────┘
    #
    def __get_pupil_positions(self, frame_bgr : np.ndarray, frame_gray : np.ndarray) -> tuple[bool, list]:
        pupils = [0.0, 0.0]

        detector = dlib.get_frontal_face_detector()
        faces = detector(frame_gray)
        if len(faces) == 0:
            return False, pupils

        face = faces[0]
        fx   = face.left() 
        fy   = face.top()
        fw   = face.right()  - fx
        fh   = face.bottom() - fy

        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        landmarks = predictor(frame_gray, face)
        
        eye_landmarks_left  = np.zeros(shape=(6, 2), dtype = int)
        eye_landmarks_right = np.zeros(shape=(6, 2), dtype = int)

        for i, idx in enumerate(range(36, 42)):
            p = landmarks.part(idx)
            eye_landmarks_left[i] = [p.x, p.y]

        for i, idx in enumerate(range(42, 48)):
            p = landmarks.part(idx)
            eye_landmarks_right[i] = [p.x, p.y]

        def get_eye_rect(eye_landmarks : np.ndarray) -> list:
            x1 = np.min(eye_landmarks[:, 0])
            y1 = np.min(eye_landmarks[:, 1])
            x2 = np.max(eye_landmarks[:, 0])
            y2 = np.max(eye_landmarks[:, 1])
            return [x1, y1, x2, y2]

        eye_rect_left  = get_eye_rect(eye_landmarks_left)
        eye_rect_right = get_eye_rect(eye_landmarks_right)
        narrowing      = cv2.getTrackbarPos("Narrowing",   "Frame")
        threshold      = cv2.getTrackbarPos("Threshold",   "Frame")
        blur           = cv2.getTrackbarPos("Blur",        "Frame")
        width_scale    = cv2.getTrackbarPos("Width Scale", "Frame")

        def show_eye_grey(title, eye_rect):
            x1, y1, x2, y2 = eye_rect
            image          = frame_gray[y1:y2, x1:x2]
            image          = cv2.resize(image, None, fx=4.0, fy=4.0)
            cv2.imshow(title, image)

        show_eye_grey("Eye Left (grey)",  eye_rect_left)
        show_eye_grey("Eye Right (grey)", eye_rect_left)

        def get_eye_masked(eye_rect, eye_landmarks):
            x1, y1, x2, y2 = eye_rect
            w              = x2 - x1
            h              = y2 - y1
            image          = frame_gray[y1:y2, x1:x2]
            mask           = np.zeros((h, w), np.uint8)
            landmarks      = np.zeros((6, 2), np.int32) # Assert for int32 in drawing.cpp:2396

            _, image  = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
            b = blur * 2 - 1
            if b > 0:
                image = cv2.medianBlur(image, b)

            for i, p in enumerate(eye_landmarks):
                px = p[0] - x1
                py = p[1] - y1

                if i == 0:
                    px += narrowing
                if i == 1 or i == 2:
                    py += narrowing
                if i == 3:
                    px -= narrowing
                if i == 4 or i == 5:
                    py -= narrowing

                px = max(0, min(px, w - 1))
                py = max(0, min(py, h - 1))
                landmarks[i] = [px, py]

            cv2.fillPoly(mask, pts=[landmarks], color=255)
            mask  = np.invert(mask)
            image = np.add(image, mask)
            return image

        eye_img_left  = get_eye_masked(eye_rect_left,  eye_landmarks_left)
        eye_img_right = get_eye_masked(eye_rect_right, eye_landmarks_right)

        def determine_x(eye_img, eye_rect, eye_landmarks) -> tuple[bool, int]:
            x1, y1, x2, y2 = eye_rect
            w              = x2 - x1
            h              = y2 - y1
            pl             = eye_landmarks[0]
            pr             = eye_landmarks[3]

            def lerp(v0, v1, t):
                return [x0 + (x1 - x0) * t for x0, x1 in zip(v0, v1)]

            def walk(a, b, steps) -> tuple[bool, int, int]:
                for i in range(0, steps):
                    p   = lerp(a, b, float(i) / float(steps))
                    px  = int(p[0] - x1)
                    py  = int(p[1] - y1)
                    px  = max(0, min(px, w - 1))
                    py  = max(0, min(py, h - 1))

                    if eye_img[py,px] == 0:
                        return True, px, py

                return False, 0, 0               

            succ_lr, plx, ply = walk(pl, pr, 20)
            succ_rl, prx, pry = walk(pr, pl, 20)

            if not succ_lr or not succ_rl:
                return False, 0

            pos_x = int((plx + prx) / 2.0)
            return True, pos_x

        # Show us where the face is
        cv2.rectangle(frame_bgr, (fx,fy), (fx+fw,fy+fh), (0, 255, 0), 3)

        def show_eye(title, eye_img):
            eye_img = cv2.resize(eye_img, None, fx=4.0, fy=4.0)
            cv2.imshow(title, eye_img)

        # Show us the masked-out eyes in grey scale
        show_eye("Eye Left",  eye_img_left)
        show_eye("Eye Right", eye_img_right)

        # Show us where the eye landmarks are
        for p in np.concatenate([eye_landmarks_left, eye_landmarks_right]):
            cv2.circle(frame_bgr, p, 2, (0, 255, 0), 2)

        succ_l, left_x  = determine_x(eye_img_left,  eye_rect_left,  eye_landmarks_left)
        succ_r, right_x = determine_x(eye_img_right, eye_rect_right, eye_landmarks_right)

        width_scale = width_scale / 100.0

        x1, y1, x2, y2  = eye_rect_left
        pupils[0] = (left_x / (x2 - x1)) * width_scale

        x1, y1, x2, y2  = eye_rect_left
        pupils[1] = (right_x / (x2 - x1)) * width_scale

        return succ_l and succ_r, pupils
