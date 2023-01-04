import cv2
import numpy as np

# This class needs to implement 3 methods:
# - init
# - shutdown
# - get_screen_position
class Eye:
    video         = None
    cascade_face  = None
    cascade_eye   = None


    # Does not take any arguments
    def init(self) -> bool:
        assert self.video    == None

        try:
            # For me, my webcam was at index 1, but it's usually 0
            self.video = cv2.VideoCapture(0)
        except:
            print("video initialization failed!")
            return False

        try:
            self.cascade_face = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
        except:
            print("cascade_face initialization failed!")
            return False

        try:
            self.cascade_eye = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
        except:
            print("cascade_eye initialization failed!")
            return False

        assert self.video         != None
        assert self.cascade_face  != None
        assert self.cascade_eye   != None

        def on_changed(value):
            pass

        cv2.namedWindow('Frame')
        cv2.createTrackbar("Threshold", "Frame", 0, 255, on_changed)
        cv2.createTrackbar("Top Cut",   "Frame", 0, 100, on_changed)
        cv2.createTrackbar("Erode",     "Frame", 0,  10, on_changed)
        cv2.createTrackbar("Dilate",    "Frame", 0,  10, on_changed)
        cv2.createTrackbar("Blur",      "Frame", 0,  10, on_changed)

        return self.__calibrate()


    # Does not take any arguments
    def shutdown(self):
        assert self.video != None

        self.video.release()
        cv2.destroyAllWindows()


    # Returns the position on a virtual screen of where the participant is looking  
    # at in normalized 0-1 2D space, with the origin being at the top left
    #
    #   0,0      1,0
    #    ┌────────┐
    #    │        │
    #    │        │
    #    └────────┘
    #   0,1      1,1
    #
    def get_screen_position(self) -> tuple[bool, float, float]:
        assert self.video != None

        success, frame_gray = self.__get_gray_frame()
        if not success:
            print("Reading frame from video source failed!")
            return False, 0, 0

        success, pupils = self.__get_pupil_positions(frame_gray)
        cv2.imshow("Frame", frame_gray)

        if success:
            assert len(pupils) == 2
            pupil_left = pupils[0]
            x = pupil_left[0]
            y = pupil_left[1]
            return True, x, y

        return False, 0, 0


    # Private method
    def __get_gray_frame(self) -> tuple[bool, np.ndarray]:
        assert self.video != None

        success, frame_bgr = self.video.read()
        if not success:
            return False, None

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        return True, frame_gray


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
    # Returns True and the two pupil positions within local eye-space 
    # or False when no face or no fitting eye pair has been found
    # 
    #   -1,-1         +1,-1     Looking at top right      bottom left
    #     ┌──────┬──────┐          ┌─────────┬──┐        ┌─┬──────────┐
    #     │      │      │          ├─────────O──┤        │ │          │
    #     ├──── 0,0 ────┤          │         │  │        │ │          │
    #     │      │      │          │         │  │        ├─O──────────┤
    #     └──────┴──────┘          └─────────┴──┘        └─┴──────────┘
    #   -1,+1         +1,+1 
    #
    def __get_pupil_positions(self, frame_gray : np.ndarray) -> tuple[bool, list]:
        pupils = [(0.0, 0.0), (0.0, 0.0)]

        faces = self.cascade_face.detectMultiScale(frame_gray)
        if len(faces) == 0:
            return False, pupils
        
        fx, fy, fw, fh = faces[0]

        face_gray = frame_gray[fy:fy+fh, fx:fx+fw]
        eyes = self.cascade_eye.detectMultiScale(face_gray)

        # First, filter out all very unlikely "eyes"
        # and categorize them to left and right
        eye_candidates_left   = []
        eye_candidates_right  = []
        for eye in eyes:
            ex, ey, ew, eh = eye

            # ex and ey are in local face space
            assert ex >= 0 and ex <= fw
            assert ey >= 0 and ey <= fh

            # We can safely ignore all "eyes" on the bottom half 
            # of the face (we don't have eyes below our nose)
            if ey > (fh / 2.0):
                continue

            if ex < (fw / 2.0):
                eye_candidates_left.append(eye)
            else:
                eye_candidates_right.append(eye)

        if len(eye_candidates_left) == 0 or len(eye_candidates_right) == 0:
            return False, pupils

        eye_left   = (0.0, 0.0, 0.0, 0.0)
        eye_right  = (fw,  fh,  0.0, 0.0)

        # 1. Pick most likely left eye
        for eye in eye_candidates_left:
            ex, ey, ew, eh = eye
            lx, ly, lw, lh = eye_left

            if (ex > lx or ey > ly) and (ew > lw or eh > lh):
                eye_left = eye

        # 2. Pick most likely right eye
        for eye in eye_candidates_right:
            ex, ey, ew, eh = eye
            rx, ry, rw, rh = eye_right

            if (ex < rx or ey > ry) and (ew > rw or eh > rh):
                eye_right = eye

        assert eye_left[0]  > 0.0 and eye_left[1]  > 0.0 and eye_left[2]  > 0.0 and eye_left[3]  > 0.0
        assert eye_right[0] > 0.0 and eye_right[1] > 0.0 and eye_right[2] > 0.0 and eye_right[3] > 0.0

        # Show us where the face is
        cv2.rectangle(frame_gray, (fx,fy), (fx+fw,fy+fh), 255, 3)

        threshold  = cv2.getTrackbarPos("Threshold", "Frame")
        top_cut    = cv2.getTrackbarPos("Top Cut",   "Frame")
        erode      = cv2.getTrackbarPos("Erode",     "Frame")
        dilate     = cv2.getTrackbarPos("Dilate",    "Frame")
        blur       = cv2.getTrackbarPos("Blur",      "Frame")

        def get_pupil_pos(eye) -> tuple[bool, float, float]:
            ex, ey, ew, eh = eye

            # Eye position in global frame space
            eye_x = fx + ex
            eye_y = fy + ey

            # Show us where the eye is
            cv2.rectangle(frame_gray, (eye_x,eye_y), (eye_x+ew,eye_y+eh), 255, 2)

            eye_gray = face_gray[ey:ey+eh, ex:ex+ew]

            _, eye_gray = cv2.threshold(eye_gray, threshold, 255, cv2.THRESH_BINARY)

            # Cut off top (eyebrows)
            top_pixels  = int((top_cut / 100.0) * eh)
            eye_gray    = eye_gray[top_pixels:eh, 0:ew]

            eye_gray  = cv2.erode(eye_gray,  None, iterations=erode)
            eye_gray  = cv2.dilate(eye_gray, None, iterations=dilate)
            
            b = blur * 2 - 1
            if b > 0:
                eye_gray = cv2.medianBlur(eye_gray, b)

            detector_params = cv2.SimpleBlobDetector_Params()
            detector_params.filterByArea  = True
            detector_params.maxArea       = 1500

            detector   = cv2.SimpleBlobDetector_create(detector_params)
            keypoints  = detector.detect(eye_gray)

            cv2.imshow('Eye', eye_gray)
            if len(keypoints) > 0:
                kp = keypoints[0].pt
                kp = [int(kp[0]), int(kp[1])]
                keypoints_img = eye_gray.copy()
                keypoints_img.fill(255)

                shape = keypoints_img.shape
                kp[0] = max(0, min(kp[0], shape[0]-1))
                kp[1] = max(0, min(kp[1], shape[1]-1))

                if cv2.waitKey(1) & 0xFF == ord('r'):
                    print(kp[0], shape[0])
                    print(kp[1], shape[1])

                #cv2.drawKeypoints(keypoints_img, keypoints, keypoints_img, 0, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                fkp = [eye_x + kp[0], eye_y + top_pixels + kp[1]]
                cv2.circle(frame_gray, fkp, 4, 255, 6)
                cv2.circle(keypoints_img, kp, 4, 0, 6)
                cv2.imshow('Eye Keypoints', keypoints_img)
                return True, 0, 0

            return False, 0, 0

        lsuccess, lx, ly  = get_pupil_pos(eye_left)
        rsuccess, rx, ry  = get_pupil_pos(eye_right)
        pupils[0]         = (lx, ly)
        pupils[1]         = (rx, ry)

        return lsuccess and rsuccess, pupils
