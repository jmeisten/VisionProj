import cv2
import threading
import time
import os

# This could in theory just post the image as a np array and post it to a rest api

class myCam:
    frame = None
    frameLock = False
    cap = None
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def getFrame(self):
        self.frameLock = True
        return self.frame

    def main(self):
        while True:
            start = time.time()

            ret, img = self.cap.read()
            
            while self.frameLock == False:
                time.sleep(.05)
            cv2.imwrite("stream/image.jpg", img)
            self.frame = str(os.getcwd() + "/stream/image.jpg")
            end = time.time()
            sleepTime = (1/30) - (end - start)
            if sleepTime < 0.0:
                sleepTime = 0.0
            time.sleep(sleepTime)

    def start(self):
        thread = threading.Thread(target=self.main, daemon=True)
        thread.start()
