#project to detect if there is a face in frame


import cv2, time, os, difflib
import numpy as np
import matplotlib.pyplot as plt
from imutils.face_utils import FaceAligner
# import tensorflow as tf

# from tensorflow.keras.models import load_model
# path to face detector  
cascadePath = "face_detection.xml"
face_cascade = cv2.CascadeClassifier(cascadePath)

cap = cv2.VideoCapture(0)

while True:
    startTime = time.time()
    _, frame = cap.read()
    # x,y,z = frame.shape()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # throw it through a detector to see if there is a face
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (30,30)
    )

    # label every face found
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow("Face cam", frame)
    stopTime = time.time()
    fps = 1/(stopTime - startTime)
    print("FPS: ", fps)

    if(cv2.waitKey(1) == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()