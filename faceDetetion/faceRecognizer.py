
from telnetlib import theNULL
import dlib, cv2, os, time, sys
import numpy as np
import keyboard
import threading
import logging
import matplotlib.pyplot as plt
from imutils.face_utils import FaceAligner

#key input setup
keyPressed=""

# # Init for face detection (not rec)
# cascadePath = "face_detection.xml"
# face_cascade = cv2.CascadeClassifier(cascadePath)


# init the models pose for landmarks, faceendcoder for image encoding
# detector to get frontal faces from img
# modelfile and config are for a dnn opensource face detection model
# net is the cv2.dnn moule that is init with config files
# Anything from dlib is machine learning. DLIB is the python compiled C++ library for neural networks and other machine learning algorithms
pose_predictor=dlib.shape_predictor('config/shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(pose_predictor)
face_encoder=dlib.face_recognition_model_v1('config/dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()
modelFile = 'config/opencv_face_detector_uint8.pb'
configFile = 'config/opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# arrays for face encodings and their corris_pressedespoinding labels 
faces=[]
name=[]
trainingPath = "config/lfw/"
faceDir="config/faceFiles/"

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
# faces = np.load(faceName)
# name = np.load(labelsName)
# count is always listed in the save file name. so check faces file name and parse for length
oldCT=0
for file in os.listdir(faceDir):
    file=str(file)
    try:
        if ("face_repr" in file):
            tmp = file.split("face_repr_")[1]
            tmp = tmp.split('.')[0]
            faces = np.load(faceDir+file).tolist()
            oldCT=int(tmp)
            print("Found and loaded Face Encodings file")
        if ("labels_" in file):
            name = np.load(faceDir+file).tolist()

            tmp = str(name)
            tmp=tmp[1:-1].split(',');

            for i in range(len(tmp)):
                tmp[i]=tmp[i].replace("'",'')
                tmp[i]=tmp[i].replace(" ",'')
                name=tmp
            if name == "[]" or name == []:
                name = []
            print("Found and loaded Labels file")
    except Exception as e:
        print("Huh something fucked up")
        print(e)

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
if faces == [] or name == []:
    first = True
    reload=True
    print("Either names or encodings could not validate. Rebuild required")

if oldCT == precount and name == []:
    loadNamesOnly=True
    name=[]
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
        #     if dir in name:
        #         imCount= imCount + len(os.listdir(path))
        #         continue
        start = time.time()
        name.append(dir)
        print(dir," has ",len(os.listdir(path))," images")
        for im in os.listdir(path):
            if loadNamesOnly == True:
                name.append(dir)
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
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                try:
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.7:
                        (h,w) = img.shape[:2]
                        box = detections[0,0, i, 3:7] * np.array ([w,h,w,h])
                        (startX, startY, endX, endY) = box.astype("int")
                        r = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                
                        faceAligned = fa.align(img, gray, r)
                        landmark = pose_predictor(faceAligned,dlib.rectangle(0,0,faceAligned.shape[0],faceAligned.shape[1]))
                        face_descriptor = face_encoder.compute_face_descriptor(faceAligned, landmark, num_jitters=2)
                        faces.append(face_descriptor)
                        name.append(str(dir))
                    
                except Exception as e:
                    print("Something happend on dir ",dir)
                    print(e)
            
            # save Periodically in case of failure we can pick up
            if saveCounter >= 100:
                print("PeriodicSave")
                saveCounter=0
                faces = np.array(faces)
                name = np.array(name)
                faceName=faceDir+'face_repr_'+str(index)+'.npy'
                labelsName=faceDir+'labels'+str(index)+'.npy'
                np.save(faceName, faces)
                np.save(labelsName, name)
                # load the arrays again.
                faces = np.load(faceName).tolist()
                name = np.load(labelsName).tolist()
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
faces = np.array(faces)
name = np.array(str(name))
faceName=faceDir+'face_repr_'+str(imCount)+'.npy'
labelsName=faceDir+'labels_'+str(imCount)+'.npy'
saveName = np.array(name)
np.save(faceName, faces)
np.save(labelsName, saveName)

# load the arrays. this is redundant if you already have it built
faces = np.load(faceName)
name = np.load(labelsName)

tmp = str(name)
tmp=tmp[1:-1].split(',');

for i in range(len(tmp)):
    tmp[i]=tmp[i].replace("'",'')
    tmp[i]=tmp[i].replace(" ",'')
    name=tmp

#face detection path
cascadePath = "face_detection.xml"
face_cascade = cv2.CascadeClassifier(cascadePath)

print("init complete")



foundScores=[]
foundNames=[]
globImg=globGray=np.empty(2, dtype=int)
decoded=False
foundRect=None
foundName=""
fx=fy=fx2=fy2 =0

frontArray= []
backArray= []
def frontHalf(array):
    print()
def backHalf(array):
    print()

# Thread for decoding images for faces
def decodeImage():
    global globImg
    global foundNames
    global boxCoors
    global foundScores
    global decoded
    global foundRect
    global fx,fy,fx2,fy2
    global foundName
    print("Started Decoding")
    while True:
        try:
            if globImg.any() and foundNames == [] and foundScores == []:
                image=globImg
                try:
                    
                    h = image.shape[0]
                    w = image.shape[1]
                except:
                    continue

                image=globImg
                gray = globGray
                blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
                net.setInput(blob)
                detections = net.forward()  
                # get all of our detected faces and print out closest matches
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
                        cv2.rectangle(globImg,(startX,startY),(endX,endY),(255,0,0), 2)

                        fx = startX
                        fy = startY
                        fx2 = endX
                        fy2 = endY
                        foundRect=r
                        faceAligned = fa.align(image, gray, r)
                        landmark = pose_predictor(faceAligned,dlib.rectangle(0,0,faceAligned.shape[0],faceAligned.shape[1]))
                        face_descriptor = face_encoder.compute_face_descriptor(faceAligned, landmark, num_jitters=2)
                        score = np.linalg.norm(faces - np.array(face_descriptor), axis=1)
                        scores.append(score)
                        scores= np.argsort(score)

                        imatches = np.argsort(score)
                        foundScores = score[imatches]
                        foundNames=np.array(name)[imatches][:10].tolist()
                        foundName=str(np.array(name)[imatches][:1].tolist())
                        cv2.putText(globImg, str(np.array(name)[imatches][:1].tolist()), (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        
                        globImg = globImg
                        # print(np.array(name)[imatches].tolist())
                        # print( score.tolist())
                        decoded=True
            else:
                if decoded==False:

                    globImg=np.empty(2, dtype=int)
                time.sleep(1)
        except Exception as e:
            print("hmmm wtf")
            exc_type, exc_obj, tb = sys.exc_info()
            f = tb.tb_frame
            lineno = tb.tb_lineno
            print(e, " Line number: ", lineno)

def foundNames():
    return foundNames

cap = cv2.VideoCapture(0)
decodingThread = threading.Thread(target=decodeImage, daemon=True)
decodingThread.start()
while True:
    try:
        startTime=time.time()
        _, image = cap.read()

        globGray = gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h = image.shape[0]
        w = image.shape[1]
        # train the model if t was pressed
        # build the detection blob and put it in the models
        # TODO test with smaller 
        if keyboard.is_pressed('t'):
            continue
            foundRect=None
            foundNames=[]
            foundScores=[]
            globImg=np.empty(2, dtype=int)
            dirToLoad=input("What folder should I load? Put path in relation to current folder    ")
            print("f")
            os.makedirs(trainingPath+dirToLoad)
            for im in os.listdir(dirToLoad):
                img = cv2.imread(os.path.join(dirToLoad,im))
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                frameHeight = img.shape[0]
                frameWidth = img.shape[1]
                blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117,   123], False, False)
                net.setInput(blob)
                detections = net.forward()
                os.rename(os.path.join(dirToLoad+img),os.path.join(trainingPath,dirToLoad,img))
                for i in range(detections.shape[2]):
                    try:
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.7:
                            (h,w) = img.shape[:2]
                            box = detections[0,0, i, 3:7] * np.array ([w,h,w,h])
                            (startX, startY, endX, endY) = box.astype("int")
                            r = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                    
                            faceAligned = fa.align(img, gray, r)
                            landmark = pose_predictor(faceAligned,dlib.rectangle(0,0,faceAligned.shape[0],faceAligned.shape[1]))
                            face_descriptor = face_encoder.compute_face_descriptor(faceAligned, landmark, num_jitters=2)
                            faces.append(face_descriptor)
                            name.append(str(dir))
                        
                    except Exception as e:
                        print("Something happend on dir ",dir)
                        print(e)
            
            faces = np.array(faces)
            name = np.array(str(name))
            faceName=faceDir+'face_repr_'+str(imCount)+'.npy'
            labelsName=faceDir+'labels_'+str(imCount)+'.npy'
            saveName = np.array(name)
            np.save(faceName, faces)
            np.save(labelsName, saveName)

            # load the arrays. this is redundant if you already have it built
            faces = np.load(faceName)
            name = np.load(labelsName)

            tmp = str(name)
            tmp=tmp[1:-1].split(',');

            for i in range(len(tmp)):
                tmp[i]=tmp[i].replace("'",'')
                tmp[i]=tmp[i].replace(" ",'')
                name=tmp

        try:
            if globImg.any():
                globImg=image
        except:
            globImg=image
           
        if (cv2.waitKey(1) == ord('q')):
            print("BYE BYE")
            break
        # if decoded then update the frame info 
        # else old name and track face   
        if decoded == True:
            decoded=False
            foundRect=None
            foundNames=[]
            foundScores=[]
            endTime=time.time()
            fps = 1//(endTime - startTime)
            cv2.putText(globImg,str(fps),(5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.imshow("TEST",globImg)
        else:

            # throw IMAGE through a detector to see if there is a face
            # faces = face_cascade.detectMultiScale(
            #     gray,
            #     scaleFactor = 1.2,
            #     minNeighbors = 5,
            #     minSize = (30,30)
            # )

            # for (x, y, w, h) in faces:
            #     cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
            #     cv2.putText(image, "Detecting Now", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


            h = image.shape[0]
            w = image.shape[1]
            cv2.putText(image, foundName, (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.rectangle(image, (fx,fy),(fx2,fy2),(255,0,0),2)
            cv2.putText(image, foundName, (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            endTime=time.time()
            fps = 1//(endTime - startTime)
            cv2.putText(image,str(fps),(5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            cv2.imshow("TEST",image)
    except Exception as e:
        print("SHIT WENT SOUTH")
        print(e)
        
cap.release()
cv2.destroyAllWindows()



#TODO 
# Logging better. Maybe built in python logging>
# Make it faster (multithreaded)
# Smaller models
# select model at start 