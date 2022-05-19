from myCam import myCam
from faceDetetion.faceRecClass import FaceRecongitionModule
import time

# This will be the main launcher for the project



pollRate = 5
# Start camera
cam = myCam()
cam.start()

# start recognizer
recognizer = FaceRecongitionModule(cam)
recognizer.start()


while True:
    start = time.time()
    foundNames = []
    try:
        foundNames = recognizer.getNames()
        recognizer.setNames([])
    except Exception as e:
        foundNames = []
        print(e)
    if foundNames != []:
        print(foundNames)
    end = time.time()

    time.sleep(pollRate - (end - start))
