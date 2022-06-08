import cv2
import dlib
import numpy as np
import time
import winsound
from math import hypot

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_PLAIN

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)

    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0,255,0), 3)
    ver_line = cv2.line(frame, center_top, center_bottom, (0,255,255), 3)
    
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

while(True):
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)

        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        end = time.time()

        if blinking_ratio > 4.5:
            cv2.putText(frame, "CLOSE", (75, 250), font, 7, (255, 0, 255))
        else :
            cv2.putText(frame, "OPEN", (75, 250), font, 7, (0, 255, 0))
            start = time.time()
        print("Duration of closing eyes:%.2f秒"%(end-start))

        if (end-start) > 2 :
            cv2.putText(frame, "TIRED", (200, 325), font, 7, (0, 0, 255))
            duration = 1000
            freq = 1000
            winsound.Beep(freq, duration)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
