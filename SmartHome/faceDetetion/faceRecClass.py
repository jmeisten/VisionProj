from telnetlib import theNULL
import dlib, cv2, os, time, sys
import numpy as np
import keyboard
import threading
import logging

from sqlalchemy import false
import myCam as mc
import matplotlib.pyplot as plt
from imutils.face_utils import FaceAligner

class FaceRecongitionModule:

    started = False
    authenticated = false
    
    def authUserFound(self,val):
        self.authenticated = val
        
    def getStarted(self):
        return self.started
    
    def __init__(self, myCam):
        self.cap = myCam

        print(os.getcwd())
        # init the models pose for landmarks, faceendcoder for image encoding
        # detector to get frontal self.faces from img
        # modelfile and config are for a dnn opensource face detection model
        # net is the cv2.dnn moule that is init with config files
        # Anything from dlib is machine learning. DLIB is the python compiled C++ library for neural networks and other machine learning algorithms
        self.pose_predictor=dlib.shape_predictor(os.getcwd() + '/faceDetetion/config/shape_predictor_68_face_landmarks.dat')
        self.fa = FaceAligner(self.pose_predictor)
        self.face_encoder=dlib.face_recognition_model_v1(os.getcwd() + '/faceDetetion/config/dlib_face_recognition_resnet_model_v1.dat')
        self.detector = dlib.get_frontal_face_detector()
        modelFile = os.getcwd() + '/faceDetetion/config/opencv_face_detector_uint8.pb'
        configFile = os.getcwd() + '/faceDetetion/config/opencv_face_detector.pbtxt'
        self.net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

        trainingPath = os.getcwd() + "/faceDetetion/config/lfw/"
        faceDir= os.getcwd() + "/faceDetetion/config/faceFiles/"

        # arrays for face encodings and their corris_pressedespoinding labels 
        self.faces=[]
        self.name=[]
        print("init")
        self.foundNames = []
        #folder list 
        folderList=[];

        # loop over directory of all training images
        # see if img count has changed since last run.
        # if it hasn't assumed that all imgs are the same. Spooky I know
        precount=0
        dirCount=0
        for dir in os.listdir(trainingPath):
            path=os.path.join(trainingPath,dir)
            folderList.append(dir)
            dirCount=dirCount+1
            for im in os.listdir(path):
                precount=precount+1
        print("Count of images is: ", precount)
        print("Count of dirs is: ", dirCount)
        # load the arrays. this is redundant if you already have it built
        # self.faces = np.load(faceName)
        # self.name = np.load(labelsName)
        # count is always listed in the save file self.name. so check self.faces file self.name and parse for length
        oldCT=0
        for file in os.listdir(faceDir):
            file=str(file)
            try:
                if ("face_repr" in file):
                    tmp = file.split("face_repr_")[1]
                    tmp = tmp.split('.')[0]
                    self.faces = np.load(faceDir+file).tolist()
                    oldCT=int(tmp)
                    print("Found and loaded Face Encodings file")
                if ("labels_" in file):
                    self.name = np.load(faceDir+file).tolist()

                    tmp = str(self.name)
                    tmp=tmp[1:-1].split(',');

                    for i in range(len(tmp)):
                        tmp[i]=tmp[i].replace("'",'')
                        tmp[i]=tmp[i].replace(" ",'')
                        self.name=tmp
                    if self.name == "[]" or self.name == []:
                        self.name = []
                    print("Found and loaded Labels file")
            except Exception as e:
                print("Huh something fucked up")


        reload=False
        first=True
        loadNamesOnly=False
        if oldCT > 0:
            first = False
        if precount!=oldCT:
            print("Need to reload model: Count missmatch: ", precount,"/",oldCT)
            reload=True
            first=True
        else:
            reload=False
        if self.faces == [] or self.name == []:
            first = True
            reload=True
            print("Either names or encodings could not validate. Rebuild required")

        if oldCT == precount and self.name == []:
            loadNamesOnly=True
            self.name=[]
            print("Reloading names array")


        # THIS BUILDS THE MODEL AND ENCODINGS. LEVERAGE THIS FOR LIVE BUILDING

        # each folder file is a person
        # load the cv2 image then convert to gray scale
        # pass gray to cv2 model to find face in image
        # the n loop over the detections and look for anything with a 70% chance of a face
        # take confident detections and pass to face algner
        # pass aligned face with landmarks to encoder to build the encoder list
        index=0
        totalTime=0
        imCount=0
        saveCounter=0
        if reload==True:
            for dir in os.listdir(trainingPath):
                print("-----------------------------------------------------------")
                path=os.path.join(trainingPath,dir)
                # if first==False and loadNamesOnly==False:
                #     if dir in self.name:
                #         imCount= imCount + len(os.listdir(path))
                #         continue
                start = time.time()
                self.name.append(dir)
                print(dir," has ",len(os.listdir(path))," images")
                for im in os.listdir(path):
                    if loadNamesOnly == True:
                        self.name.append(dir)
                        index=index+1
                        imCount= imCount + 1
                        continue
                    imCount=imCount+1
                    saveCounter=saveCounter+1
                    img = cv2.imread(os.path.join(path,im))
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    frameHeight = img.shape[0]
                    frameWidth = img.shape[1]
                    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117,   123], False, False)
                    self.net.setInput(blob)
                    detections = self.net.forward()
                    for i in range(detections.shape[2]):
                        try:
                            confidence = detections[0, 0, i, 2]
                            if confidence > 0.7:
                                (h,w) = img.shape[:2]
                                box = detections[0,0, i, 3:7] * np.array ([w,h,w,h])
                                (startX, startY, endX, endY) = box.astype("int")
                                r = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                        
                                faceAligned = self.fa.align(img, gray, r)
                                landmark = self.pose_predictor(faceAligned,dlib.rectangle(0,0,faceAligned.shape[0],faceAligned.shape[1]))
                                face_descriptor = self.face_encoder.compute_face_descriptor(faceAligned, landmark, num_jitters=2)
                                self.faces.append(face_descriptor)
                                self.name.append(str(dir))
                            
                        except Exception as e:
                            print("Something happend on dir ",dir)
                            print(e)
                    
                    # save Periodically in case of failure we can pick up
                    if saveCounter >= 100:
                        print("PeriodicSave")
                        saveCounter=0
                        self.faces = np.array(self.faces)
                        self.name = np.array(self.name)
                        faceName=faceDir+'face_repr_'+str(index)+'.npy'
                        labelsName=faceDir+'labels'+str(index)+'.npy'
                        np.save(faceName, self.faces)
                        np.save(labelsName, self.name)
                        # load the arrays again.
                        self.faces = np.load(faceName).tolist()
                        self.name = np.load(labelsName).tolist()
                        print("saveDone")

                end = time.time()
                totalTime = totalTime + (end-start)
                print("Finished with person ",dir," index:", index, " of: ", len(os.listdir(trainingPath)))
                print("Time taken: " , end -start )
                print(imCount," Of ",precount," done." )
                print("Progress:", (imCount/precount) *100,"%")
                index = index +1

            print("-----------------------------------------------------------")
            avgTime = totalTime/index
            print("Total Time to build: ", totalTime)
            print("Avg time per person decoding during model build: ", avgTime)
        else:
            imCount=oldCT

        #delete all the periodic saves before saving the final
        dir = faceDir
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
        # convert the lists to np arrays and save to disk for future use so we can avoid prebuilding 
        # tricky tricky americans
        self.faces = np.array(self.faces)
        self.name = np.array(str(self.name))
        faceName=faceDir+'face_repr_'+str(imCount)+'.npy'
        labelsName=faceDir+'labels_'+str(imCount)+'.npy'
        saveName = np.array(self.name)
        np.save(faceName, self.faces)
        np.save(labelsName, saveName)

        # load the arrays. this is redundant if you already have it built
        self.faces = np.load(faceName)
        self.name = np.load(labelsName)

        tmp = str(self.name)
        tmp=tmp[1:-1].split(',');

        for i in range(len(tmp)):
            tmp[i]=tmp[i].replace("'",'')
            tmp[i]=tmp[i].replace(" ",'')
            self.name=tmp

        print("init complete")
        
    def getNames(self):
        return self.foundNames
    
    def setNames(self, name):
        self.foundNames = name

    def decodeImage(self, img):
        try:
            image=img
            if len(img)<=1:
                print("image was null")
                return
            h = image.shape[0]
            w = image.shape[1]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
            self.net.setInput(blob)
            detections = self.net.forward()  
            # get all of our detected self.faces and print out closest matches
            scores=[]
            detectedNames=[]

            #todo one forward one backward traversal simulatniously
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7 and confidence < 1.0:
                    box = detections[0,0, i, 3:7] * np.array ([w,h,w,h])
                    (startX, startY, endX, endY) = box.astype("int")
                    r = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                    # boxCoors=[int(startX), int(startY), int(endX), int(endY)]
                    # cv2.rectangle(globImg,(startX,startY),(endX,endY),(255,0,0), 2)

                    faceAligned = self.fa.align(image, gray, r)
                    landmark = self.pose_predictor(faceAligned,dlib.rectangle(0,0,faceAligned.shape[0],faceAligned.shape[1]))
                    face_descriptor = self.face_encoder.compute_face_descriptor(faceAligned, landmark, num_jitters=2)
                    score = np.linalg.norm(self.faces - np.array(face_descriptor), axis=1)
                    scores.append(score)
                    scores= np.argsort(score)

                    imatches = np.argsort(score)
                    # foundScores = score[imatches]
                    self.foundNames=np.array(self.name)[imatches][:10].tolist()
                    self.foundName=str(np.array(self.name)[imatches][:3].tolist())
        except Exception as e:
            print("Shit went south decoding faces")
            exc_type, exc_obj, tb = sys.exc_info()
            f = tb.tb_frame
            lineno = tb.tb_lineno
            print(e, " Line number: ", lineno)        

    def main(self):
        self.started = True
        self.authenticated = False
        while True:
            try:
                
                if self.authenticated == False:
                    self.authenticated = False
                    image = cv2.imread(self.cap.getFrame())
                    self.cap.frameLock = False
                    self.decodeImage(image)
                else:
                    # If we have a user lets sleep a bit and free up resources
                    # print("Taking a wee nap since we are authenticated")
                    time.sleep(1)
            except Exception as e:
                print(e)


    def start(self):
        thread = threading.Thread(target=self.main, daemon=True)
        thread.start()
