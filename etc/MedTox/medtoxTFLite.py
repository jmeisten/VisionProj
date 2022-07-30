import argparse
import os
import platform
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import tensorflow as tf
import torch.backends.cudnn as cudnn
from utils.general import xyxy2xywh, non_max_suppression, scale_coords
from utils.dataloaders import LoadImages, LoadStreams
from models.common import DetectMultiBackend
from utils.plots import Annotator, colors

ROOTDIR=Path(__file__).resolve().parents[0]

yamlFilePath="./medToxDataSet/medToxData.yaml"
tfLitePath="./models/medtox_model.tflite"
saveDir = "./medToxDataSet/output/"
DISPLAY_WINDOW_NAME="Inference output"


parser = argparse.ArgumentParser()
parser.add_argument("--source",type=str,default=".\\images\\")
args = parser.parse_args()
confidenceThresh = .50
NMSThreshold=.45
cap = cv2.VideoCapture(0)
# 
save_img = True
done = False
dataset = None
i=0
device = "cpu"
model = DetectMultiBackend(tfLitePath, device=device, dnn=False, data=yamlFilePath, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
dataset = []
cap = None

if args.source =="cam":
    cap = cv2.VideoCapture(0)
    

while done == False:
    start = time.time()
    img = None
    path = None
    if args.source == "cam":
        cudnn.benchmark = True  # set True to speed up constant image size inference
        ret, img = cap.read()
        img = cv2.resize(img,(2592,1944))
        cv2.imwrite("stream/image.jpg", img)
        dataset = LoadImages("./stream/", img_size=640, stride=stride, auto=pt)
    else:
        dataset = LoadImages(args.source, img_size=640, stride=stride, auto=pt)
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        annotator = None
        # Inference
        pred = model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, confidenceThresh, NMSThreshold, None, False, max_det=100)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        detectedItems= []
        for i, det in enumerate(pred):  # per image
            
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            bcCt = 0
            lineCt = 0
            stripCt = 0            
            smudgeCt = 0
            ctArray = [bcCt,lineCt,stripCt,smudgeCt]
            
            p = Path(p)  # to Path
            save_path = str(saveDir + p.name)  # im.jpg
            txt_path = str(saveDir + 'labels' + "/" + p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if False else im0  # for save_crop
            annotator = Annotator(im0, line_width=2, example=str(names))
            
            detectedItems=[]
            strips = []
            testLines = []
            bc = None
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # print(s)
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    
                    label = None if False else (names[c] if False else f'{names[c]} {conf:.2f}')
                    
                    p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    center = (p1[0] + (p2[0]-p1[0])/2, p1[1] + (p2[1]-p1[1])/2)
                    
                    tLabel = label.split(" ")[0]
                    boxOBJ = {
                        "p1":p1,
                        "p2":p2,
                        "center":center,
                        "class":tLabel
                    }
                    detectedItems.append(boxOBJ)
                    if label.split(" ")[0] == "testLine":
                        lineCt+=1
                        testLines.append(boxOBJ)

                    elif label.split(" ")[0] == "barcode":
                        bcCt +=1
                        bc = boxOBJ
                        
                    elif label.split(" ")[0] == "strip":
                        stripCt +=1
                        strips.append(boxOBJ)
                    
                    # This will end up being to see if the box will be inside or surrounding one. 
                    # Do this if we start getting a bunch of extra shit showing up that the confidence tune wont fix
                    # DO NOT PUT CONF OVER 55 for now
                    if False:

                        insideBox = False
                        surroundingBox = False
                        tl = p1
                        tr = (p2[0] ,p1[1])
                        br = p2
                        bl = (p1[0],p2[1])
                        
                        points = [tl,tr,br,bl]
                        for box in self.drawnBoxes:
                            c = box["class"]
                            if c == tLabel:
                                # Check if box will be inside another box of same class
                                for point in points:
                                    if self.pointInBox(point,box):
                                        insideBox = True

                                btr = (box["p1"][0], box["p1"][1])
                                bbl = (box["p1"][0], box["p2"][1])
                                btl = (box["p2"][0], box["p1"][1])
                                bbr = (box["p1"][0], box["p2"][1])
                                boxPoints = [btl,btr,bbr,bbl]
                                # Check if box will engulf another box of same class
                                for point in boxPoints:
                                    if self.pointInBox(point,boxOBJ):
                                        surroundingBox = True

                    if True:  # Write to file
                        annotator.box_label(xyxy, tLabel, color=colors(c, True))
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if True else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')


            stripsWithAssociatedLines = []
            si = 0
            try:
                for strip in strips:
                    linesInStrip = []
                    for line in testLines:
                        if annotator.pointInbetweenPoints(line["center"],strip["p1"],strip["p2"]):
                            linesInStrip.append(line)
                    
                    # TODO: Sort the lines in the strip based on the location of the barcode (left or right)
                    sortedLineInStrip = sorted(linesInStrip,key=lambda d: d["center"][0])
                    sortedLineInStrip[0]["class"] = "ctrlLine"
                    i=1
                    while i < len(sortedLineInStrip):
                        sortedLineInStrip[i]["class"] = "passedTest"
                        i+=1

                    finishedAddingNegs = False
                    i=0
                    negAdded = False
                    while not finishedAddingNegs:
                        while i+1 < len(sortedLineInStrip) and not 5 == len(sortedLineInStrip):
                            distanceBetweenTests =  sortedLineInStrip[i+1]["center"][0] - sortedLineInStrip[i]["center"][0]
                            if distanceBetweenTests > 115 :
                                print("Distance between tests on strip {} was further than expected: {} between tests {} and {}".format(si, distanceBetweenTests, i, i+1))
                                p1 = (sortedLineInStrip[i]["p1"][0] + 100, sortedLineInStrip[i]["p1"][1])
                                p2 = (sortedLineInStrip[i]["p2"][0] + 100, sortedLineInStrip[i]["p2"][1])
                                center = (sortedLineInStrip[i]["center"][0]+100,sortedLineInStrip[i]["center"][1])
                                negTest = {
                                    "p1":p1, 
                                    "p2":p2,
                                    "center":center,
                                    "class":"negativeTest"
                                }
                                sortedLineInStrip.append(negTest)
                                xyxy = [p1[0],p1[1],p2[0],p2[1]]
                                annotator.box_label(xyxy, "failedTest", color=colors(5, True))
                                negAdded = True
                            i+=1
                        if (negAdded):
                            negAdded = False
                            print("neg added")
                            sortedLineInStrip = sorted(sortedLineInStrip,key=lambda d: d["center"][0])
                            i=0
                        else:
                            finishedAddingNegs = True
                            i = len(sortedLineInStrip)
                            print(i)
                            while i <=4:
                                p1 = (sortedLineInStrip[i-1]["p1"][0] + 90, sortedLineInStrip[i-1]["p1"][1])
                                p2 = (sortedLineInStrip[i-1]["p2"][0] + 90, sortedLineInStrip[i-1]["p2"][1])
                                center = (sortedLineInStrip[i-1]["center"][0]+90,sortedLineInStrip[i-1]["center"][1])
                                negTest = {
                                    "p1":p1, 
                                    "p2":p2,
                                    "center":center,
                                    "class":"failedTest"
                                }
                                sortedLineInStrip.append(negTest)
                                xyxy = [p1[0],p1[1],p2[0],p2[1]]
                                annotator.box_label(xyxy, "failedTest", color=colors(8, True))
                                i+=1
                            
                    sortedLineInStrip = sorted(sortedLineInStrip,key=lambda d: d["center"][0])
                    i = len(sortedLineInStrip)-1
                    si+=1                    
                
                    stripWithLine = {
                        "p1":strip["p1"],
                        "p2":strip["p2"],
                        "center":strip["center"],
                        "class":strip["class"],
                        "lines":sortedLineInStrip
                    }

                    stripsWithAssociatedLines.append(stripWithLine)
            except Exception as e:
                print(e)

            sortedList = sorted(stripsWithAssociatedLines,key=lambda d: d["center"][1])

            # Stream results
            im0 = annotator.result()
            # print("Found {} objects and drew those bitches".format(len(annotator.getBoxLocationsDrawn())))
            # print(annotator.getBoxLocationsDrawn())
            im0 = cv2.putText(im0, 'Barcodes: {}'.format(bcCt), (10,15), 0, 
                1, colors(0,True), 1, cv2.LINE_AA)
            im0 = cv2.putText(im0, 'Strips Seen: {}'.format(stripCt), (10,45), 0, 
                1, colors(1,True), 1, cv2.LINE_AA)
            im0 = cv2.putText(im0, 'Test Lines Seen: {}'.format(lineCt), (10,75), 0, 
                1, colors(2,True), 1, cv2.LINE_AA)

            i = 0
            for item in sortedList:
                print()
                print("Strip {} \nLines found in strip {}\n{}".format(i+1,len(item["lines"]),item))
                print()
                i+=1
            stop = time.time()
            fps = (stop - start)
            im0 = cv2.putText(im0, "Decode time: {:.2F}".format(fps), (im0.shape[0]-30,30), 0, 
                1, colors(4,True), 1, cv2.LINE_AA)

            im0 = cv2.resize(im0,(1920,1080))
            cv2.imshow(DISPLAY_WINDOW_NAME, im0)
                
            t = 1 if args.source == "cam" else 0
            
            if(cv2.waitKey(t) == ord('q')):
                done = True
                cv2.destroyAllWindows()
                quit()

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)
