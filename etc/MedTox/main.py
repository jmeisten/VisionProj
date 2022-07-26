from linecache import getline
from re import L
import time
import cv2
from pyzbar import pyzbar
import imutils
import os
from matplotlib.ft2font import LOAD_NO_RECURSE
import numpy as np

class ROI:
    topLeft=(0,0)
    bottomRight=(0,0)
    def __init__(self):
        pass

class LineObj:
    line=None
    coord1=(0,0)
    coord2=(0,0)
    
    def __init__(self,l,c1,c2):
        self.line = l
        self.coord1=c1
        self.coord2=c2

class Barcode:
    data=None
    bcType=None
    x1=None
    y1=None
    x2=None
    y2=None
    def __init__(self, d, TL,BR,bType):
        self.data = d
        self.x1 = TL[0]
        self.y1 = TL[1]
        self.x2 = BR[0]
        self.x2 = BR[1]
        self.bctype = bType

imgDir ="./MedToxPhotos/"
failPath='./Failed/'
passPath='./Pass/'
passCT=0
failCt=0
thresholdVal = 175
dim = (683,384) # half of my screen size of 1366x768
vertLinesWanted = True


def scanBarCode(img):
    barcode = None
    objects = pyzbar.decode(img)
    print(len(objects))
    for obj in objects:
        print("Detected barcode: ", obj)
        print("Data: {}".format(obj[0]))
        print("Data: {}".format(obj[1]))
        print("Data: {}".format(obj[2]))
        print("Data: {}".format(obj[3]))
        barcode = Barcode(obj[0],obj[2],obj[3],obj[1])
    
    return barcode

# TODO
# This will give us our regions of interest in each lane.
# My hope is that by breaking down the images into regions 
#   we can speed up the detection of lines and avoid extra crap 
#   from being accidentally analyzed.
def createRegionsOfInterest(gray,img,thresh):
    lanes=[]
    scanBarCode(img)
    #  # lets find our lines bby!
    # start = time.time()
    # edges = cv2.Canny(thresh,50,thresholdVal,apertureSize = 3,L2gradient=True)
    # minLineLength = 150
    # maxLineGap = 50
    # theta=0
    # if vertLinesWanted:
    #     theta=np.pi/128
    # else:
    #     theta=np.pi/128

    # lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/128, threshold=10,lines=np.array([]), minLineLength=minLineLength,maxLineGap=maxLineGap)
    # # draw them bby!
    # a,b,c = lines.shape
    # for i in range(a):
    #     l = lines[i][0]
    #     Xo=l[0]
    #     Yo=l[1]
    #     Xa=l[2]
    #     Ya=l[3]
    #     cv2.line(img, (Xo,Yo), (Xa,Ya), (0, 0, 255), 3, cv2.LINE_AA)
    #     # cv2.imwrite('houghlines5.jpg',img)
    # end = time.time()
    # print("Line finding took {}s".format(end-start))
    # return LOAD_NO_RECURSE

# Lets fix the image so its not cock eyed so we can do detection better
def rectifyImage(img,vert):
    start = time.time()
    sens = 80
    WHITE_LOWER = np.array([50, 50, 30], dtype ="uint8")
    WHITE_UPPER = np.array([255, 255, 255], dtype ="uint8")
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,WHITE_LOWER,WHITE_UPPER)

    img = cv2.bitwise_and(img,img,mask=mask)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
   

    edges = cv2.Canny(thresh,100,200,apertureSize = 3,L2gradient=True)
    minLineLength = 400
    maxLineGap = 50
    theta=0
    if vertLinesWanted:
        theta=np.pi/128
    else:
        theta=np.pi/128
    
    # params are image source, rho accuracy, theta accuracy, threshold for a line (should be the min length)
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/500, threshold=40,lines=np.array([]), minLineLength=minLineLength,maxLineGap=maxLineGap)
    # draw them bby!
    a,b,c = lines.shape
    highestLine = None
    for i in range(a):
        l = lines[i][0]
        Xo=l[0]
        Yo=l[1]
        Xa=l[2]
        Ya=l[3]
        
        print("Y: {}".format(Ya-Yo))
        print("X: {}".format(Xa-Xo))
        cv2.line(img, (Xo,Yo), (Xa,Ya), (0, 0, 255), 3, cv2.LINE_AA)

    end = time.time()
    cv2.imshow('rectTreash',thresh)
    print("Line finding took {}s".format(end-start))
    end=time.time()
    print("Image Rectification took {}s".format(end-start))
    return img

def getLinesAndApplyToImage(img,thresh,vertLinesWanted):
     # lets find our lines bby!
    start = time.time()
    edges = cv2.Canny(thresh,50,thresholdVal,apertureSize = 3,L2gradient=True)
    minLineLength = 20
    maxLineGap = 4
    theta=0
    if vertLinesWanted:
        print("here")
        theta=np.pi/128
    else:
        print("there")
        theta=np.pi/128

    lines = cv2.HoughLinesP(image=edges,rho=.2,theta=theta, threshold=10,lines=np.array([]), minLineLength=minLineLength,maxLineGap=maxLineGap)
    # draw them bby!
    a,b,c = lines.shape
    for i in range(a):
        l = lines[i][0]
        Xo=l[0]
        Yo=l[1]
        Xa=l[2]
        Ya=l[3]
        if (vertLinesWanted and Yo-Ya!=0):
            cv2.line(img, (Xo,Yo), (Xa,Ya), (0, 0, 255), 3, cv2.LINE_AA)
        # cv2.imwrite('houghlines5.jpg',img)
    end = time.time()
    print("Line finding took {}s".format(end-start))
    return img

# rois = createRegionsOfInterest()
cv2.namedWindow("BinaryImage")
cv2.namedWindow("OG img")
for imgPath in os.listdir(imgDir):
    start = time.time()
    img = cv2.imread(imgDir+imgPath)
    imPure = cv2.imread(imgDir+imgPath)
    # masked=rectifyImage(img,vertLinesWanted)

    #take im and convert to gray then 
    # do a binary threshold, any pixel with a value over thresholdValue and set those white
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # gausian hpf and butterworth filters
    if (False):
        # startFilter = time.time()
        # F = np.fft.fft2(gray)
        # Fshift = np.fft.fftshift(F)

        # # low pass butterworth

        # M = gray.shape[0]
        # N = gray.shape[1]
        # H = np.zeros((M,N), dtype=np.float32)
        # D0 = 10 #cutoff frequency
        # n = 25 # order

        # for u in range(M):
        #     for v in range(N):
        #         D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
        #         H[u,v]= 1 / ( 1 + (D/D0)**(2*n))
        # lGShift= Fshift*H
        # lG = np.fft.ifftshift(lGShift)
        # lg = np.abs(np.fft.ifft2(lG))
        # cv2.imshow("Low Pass Filter",lg)

        # # high pass butterworth
        # M = gray.shape[0]
        # N = gray.shape[1]
        # HPF = np.zeros((M,N), dtype=np.float32)
        # D0 = 360
        # n = 25
        # for u in range(M):
        #     for v in range(N):
        #         D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
        #         HPF[u,v]= 1 / ( 1 + (D/D0)**(2*n))

        # Gshift = Fshift*HPF #(or HPF)
        # G = np.fft.ifftshift(Gshift) 
        # g = np.abs(np.fft.ifft2(G))

        #gaussian hpf
        imCp= cv2.resize(imPure, dim)
        # 127 to gray the img
        # hpf = imCp -cv2.GaussianBlur(imCp,(21,21),3)+127

    #lapacian and sobel hpf
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    laplace = cv2.Laplacian(gray,cv2.CV_64F)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=7)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=7)
    sobel = cv2.bitwise_and(sobelx,sobely)

    # have to do this or thresholding breaks due to dimensions bullshit I don't feel like fixing rn
    cv2.imwrite("test.jpg",sobelx)
    im2 = cv2.imread("test.jpg")
    gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY) 

    ret, thresh = cv2.threshold(gray,thresholdVal,255,cv2.THRESH_BINARY)
    createRegionsOfInterest(gray,img,thresh)
    img = getLinesAndApplyToImage(gray,thresh,vertLinesWanted)
    _, thresh = cv2.threshold(gray,thresholdVal,255,cv2.THRESH_BINARY)
    im2 = getLinesAndApplyToImage(im2,thresh,vertLinesWanted)
    end = time.time()

    threshingTime = end - start

    print("Done thresholding. Thresholding took {}s.\nFinding and drawing lines".format(threshingTime))
    
    # stopFilter = time.time()
    # print("High Pass filtering took {}s".format(stopFilter-startFilter))
    # cv2.imshow("High Pass Filter",g)

    threshS = cv2.resize(thresh, dim)
    # im2s = cv2.resize(im2, dim)
    imgS = cv2.resize(img, dim)
    im2 = cv2.resize(im2, dim)
    imgPureS = cv2.resize(imPure, dim)
    # cv2.imshow("Rectified image",masked)
    # cv2.imshow('BinaryImage', threshS)
    cv2.imshow("OG img", imgPureS)
    cv2.imshow("Lined Image", imgS)
    cv2.imshow("Sobel Linex Image", im2)
    key = cv2.waitKey(0)
    print(key)
    if key == 113:
        cv2.destroyAllWindows()
        break
    elif key == 114:
        failCt+=1
        cv2.imwrite(failPath+imgPath,imPure)
    elif key == 115:
        passCT+=1
        cv2.imwrite(passPath+imgPath,imPure)


class ROI:
    topLeft=(0,0)
    bottomRight=(0,0)
    def __init__(self):
        pass

class LineObj:
    line=None
    coord1=(0,0)
    coord2=(0,0)
    
    def __init__(self,l,c1,c2):
        self.line = l
        self.coord1=c1
        self.coord2=c2

class Barcode:
    data=None
    bcType=None
    x1,y1,x2,y2=None
    def __init__(self, d, TL,BR,bType):
        self.data = d
        self.x1 = TL[0]
        self.y1 = TL[1]
        self.x2 = BR[0]
        self.x2 = BR[1]
        self.bctype = bType