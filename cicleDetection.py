from typing import List
import cv2
import time 
import numpy as np
import logging
import threading
from datetime import datetime


# if using logitech c270 or other yuv format use this for converting to bgr
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# other wise use this
# gary = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Hough Circles Parameters
minDist = 1000
pr1 = 255
pr2 = 10
minR = 60
maxR = 160

# globals 
image = None
circle = None

frameRate = 1/30

# list for analysis of detection thread
avgTimeList = list()

def sleep(t):
    time.sleep(t)

# finds HoughCircles on a grayscale frame and returns 2darray of [[x,y,r]]
# this will turn to object detection eventually (with the coral accelerator)
def detect(frame):
    start = time.time()

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist, param1=pr1, param2=pr2, minRadius=minR, maxRadius=maxR)
    circles = np.round(circles[0, :]).astype("int")
    end = time.time()
    avgTimeList.append(end-start)
    return circles

# pull the most recent frame and null out that object
# this will turn to object detection thread 
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
            
# TODO: turn this into __main__ so it looks nicer  

# create camera using usb webcams
cam = cv2.VideoCapture(0)
while cam.isOpened() :
    # spawn a daemon thread to just do circle detection (dies with program)
    circleThread = threading.Thread(target=detectCirclesThread, daemon=True)
    circleThread.start()
    sleep(3)
    while True: 
        start = time.time()

        # get image and grayscale it before setting it as global img
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,5)

        # create copy of image to show at end
        output = gray.copy()
        if image is None:
            image = gray

        # quit if q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # get most recent output from circle detection thread and draw onto copied image
        if not circle is None:
            for (x,y,r) in circle:
                cv2.circle(output, (x,y),r, (0,255,0),10)
        # side by side of images with circles on it 
        cv2.imshow("Before",  np.hstack([gray,output]))
        circle = None

        # this will keep up at ~30 fps since thats what Logic C270 runs at
        elapsed = time.time() - start
        sleepTime = max(frameRate - elapsed,0)
        sleep(sleepTime)

    cam.release()
    cv2.destroyAllWindows()
    totalTime = 0
    for i in range(len(avgTimeList)):
        totalTime = totalTime + avgTimeList[i]
    avgTime = totalTime / len(avgTimeList)

    print("average hough calculation time is ",  avgTime)
