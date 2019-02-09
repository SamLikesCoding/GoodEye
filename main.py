''' Required modules'''
import cv2
from imutils import paths
import numpy as np
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import dlib
import tkinter as tk
import tkinter.messagebox as tkMSG
import time
import random

cam = cv2.VideoCapture(0)
threshold_ST = 2
width = 450
m_state = "inactive"
warnings = {
    1: " Are you okay, looks like you're tired",
    2: " Looks like you're stressed out, take a break",
    3: " You look tired, get up from chair and exercise a bit",
    4: " You look sleepy, i recommend you to take a nap",
    5: " You look really tired, i recommend you to pause the work and take a nap"
}

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0
mins = 0
ALERT = False

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def monitor():
    while True:
        ret, frame = cam.read()
        frame = imutils.resize(frame, width=width)
        grayf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(grayf, 0)

        for r in rects:
            shape = predictor(grayf, r)
            shape = face_utils.shape_to_np(shape)
            lE = shape[lStart:lEnd]
            rE = shape[rStart:rEnd]
            lEar = eye_aspect_ratio(lE)
            rEar = eye_aspect_ratio(rE)
            # Average ear ratio
            Ear = (lEar + rEar) / 2.0
            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(lE)
            rightEyeHull = cv2.convexHull(rE)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if Ear < EYE_AR_THRESH:
                COUNTER += 1

                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALERT:
                        print("Log: Alert")
                        tkMSG.showwarning(" Eye Health warning ", rand_warn())
                        ALERT = True
                else:
                    COUNTER = 0
                    ALERT = False
        cv2.imshow("Eye Frame", frame)


def set_st(t):
    threshold_ST = t


def on_close_event():
    cv2.destroyAllWindows()
    cam.release()
    exit(0)

def rand_warn():
    return warnings[random.randint(1,6)]


main_app = tk.Tk()
#main_app.attributes("-fullscreen", 1)
title_frame = tk.LabelFrame(main_app, text="Console")
title_frame.pack(fill="both", expand="yes")
title = tk.Label(title_frame, text="GoodEYE : a reminder to your eyes")
title.pack()
status = tk.Label(title_frame, text="Eye monitor is "+m_state)
status.pack(side=tk.BOTTOM)
#out_frame = tk.Canvas(title_frame, width=cam.get(cv2.CAP_PROP_FRAME_WIDTH), height=cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
main_frame = tk.Frame(main_app)
main_frame.pack(side=tk.BOTTOM)
scr_time_frame = tk.Frame(main_frame)
scr_time_frame.pack()
'''

scr_time_label = tk.Label(scr_time_frame, text="Screen Time (default : 20 mins)")
scr_time_label.pack(side=tk.LEFT)
scr_time_entry = tk.Entry(scr_time_frame, bd=6)
scr_time_entry.pack(side=tk.RIGHT)
set_scr_time = tk.Button(scr_time_frame, text=" Set Screen Time", command=set_st(int(scr_time_entry.get())))
set_scr_time.pack(side=tk.BOTTOM)
'''

exit_button = tk.Button(main_frame, text="Exit", command=on_close_event)
exit_button.pack(side=tk.BOTTOM)


'''
def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)

    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    return cv2.minAreaRect(c)


def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth
'''

if __name__ == "__main__":
    #scr_time_entry.insert(10,"20")
    main_app.mainloop()
    while mins <= threshold_ST:
        mins += 1
        time.sleep(1)
    m_state = "active"
    status.config(text="Eye monitor is "+m_state)
    status.pack(side=tk.BOTTOM)
    monitor()