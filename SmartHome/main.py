from timeit import Timer
from myCam import myCam
from faceDetetion.faceRecClass import FaceRecongitionModule
from utils.Timer import Timer
from gestureDetection.GestureRecognizer import HandGestureModule
import time
import cv2
import sys
def getAuthorizedUsers():
    #TODO: Load from file at init
    rVal = ["Jake","Jake_Meisten","Jacob_Meisten"] 
    return rVal

# This will be the main launcher for the project



fpsTarget = 15
# Time period where gestures are allowed
authorizeWindow = 15
# Rate at which image refreshes
pollRate = 1/(fpsTarget)

authUserFound = False

# Start camera
cam = myCam()
cam.start()
print("Starting video capture")
time.sleep(3)
# start recognizer
recognizer = FaceRecongitionModule(cam)
recognizer.start()

handGestureRecognizer = HandGestureModule(cam)
handGestureRecognizer.start()

authTimer = Timer(authorizeWindow)
print("Waiting for modules to start")
finished = False
while finished == False:
    finished = (recognizer.getStarted() and handGestureRecognizer.getStarted())

authedName = ""
foundNames = []
timerNotExpired = False
while True:
    start = time.time()
    if authTimer.isExpired == False and not authedName == "":
        timerNotExpired = True
        # Found an authorized user Check gestures
        gesturesFound = handGestureRecognizer.getGestures()
        print(gesturesFound)
        recognizer.authUserFound(True)
    else:
        authedName = ""
        if timerNotExpired:
            timerNotExpired = False
            authedName = ""
            print("Authorization window has ended. Searching for new faces")
            recognizer.authUserFound(False)
            
        # We have not found a registered user yet so we have to wait until one is found
        try:
            foundNames = recognizer.getNames()
        except Exception as e:
            foundNames = []
            print(e)
        if foundNames == []:
            continue
        print("Found Names: {}".format(foundNames))
        for x in foundNames:
            if x in getAuthorizedUsers():
                authTimer.start(time.time())
                authedName = x
                handGestureRecognizer.authenticated = True
        if authedName != "":
            print("Authorizing on user: {}".format(authedName))            
    
    end = time.time()
    sleepTime = pollRate - (end - start)
    if sleepTime < 0.0:
        sleepTime = 0.0
    time.sleep(sleepTime)
