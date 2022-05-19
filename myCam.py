import cv2
import threading
import time
import os

class myCam:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def getFrame(self):
        return self.frame

    def main(self):
        while True:
            start = time.time()

            ret, img = self.cap.read()
            
            self.frame = str(os.getcwd() + "/stream/image.jpg")
            cv2.imwrite("stream/image.jpg", img)
            end = time.time()

            time.sleep((1/30) - (end - start))

    def start(self):
        thread = threading.Thread(target=self.main, daemon=True)
        thread.start()
