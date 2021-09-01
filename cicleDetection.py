from typing import List
import cv2
import time 
import numpy as np
import logging
import threading
from datetime import datetime


# if using logitech c270 or other yuv format use this for converting to bgr
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Hough Circles Parameters
minDist = 1000
pr1 = 255
pr2 = 10
minR = 60
maxR = 160

image = None
circle = None

frameRate = 1/30

cam = cv2.VideoCapture(0)

avgTimeList = list()

def sleep(t):
    time.sleep(t)

def detect(frame):
    start = time.time()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray,5)
    # print(len(gray.shape))

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist, param1=pr1, param2=pr2, minRadius=minR, maxRadius=maxR)
    circles = np.round(circles[0, :]).astype("int")
    end = time.time()
    print("CircleTime = ", end - start)
    print("Circle ", circles)
    avgTimeList.append(end-start)
    return circles

def detectCirclesThread():
    while 1:
        global image
        global circle
        if not image is None:
            frame = image
            image = None
            circle_temp = detect(frame)

            if circle is None:
                circle = circle_temp
            
            
while cam.isOpened() :
    # circleDetecting thread
    circleThread = threading.Thread(target=detectCirclesThread, daemon=True)
    sleep(3)
    circleThread.start()
    while True: 
        start = time.time()

        # get image and grayscale it before setting it as global img
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,5)
        output = gray.copy()
        if image is None:
            image = gray
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not circle is None:
            for (x,y,r) in circle:
                cv2.circle(output, (x,y),r, (0,255,0),10)
        cv2.imshow("Before",  np.hstack([gray,output]))
        circle = None
        elapsed = time.time() - start
        sleepTime = max(frameRate - elapsed,0)
        sleep(sleepTime)

    cam.release()
    cv2.destroyAllWindows()
    totalTime = 0
    for i in range(len(avgTimeList)):
        totalTime = totalTime + avgTimeList[i]
    avgTime = totalTime / len(avgTimeList)

    print("average circletime is ",  avgTime)
