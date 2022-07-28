import argparse
import os
import platform
import sys
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
    img = None
    path = None
    if args.source == "cam":
        cudnn.benchmark = True  # set True to speed up constant image size inference
        _, img = cap.read()
        cv2.imwrite("./stream/image.jpg", img)
        dataset = LoadImages("./stream/", img_size=640, stride=stride, auto=pt)
    else:
        dataset = LoadImages(args.source, img_size=640, stride=stride, auto=pt)
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, confidenceThresh, NMSThreshold, None, False, max_det=100)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
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
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                print(s)
                for *xyxy, conf, cls in reversed(det):
                    if True:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if True else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img :  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if False else (names[c] if False else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if label.split(" ")[0] == "testLine":
                            lineCt+=1

                        if label.split(" ")[0] == "barcode":
                            bcCt +=1
                            
                        if label.split(" ")[0] == "strip":
                            stripCt +=1
                    
            # Stream results
            im0 = annotator.result()
            # print("Found {} objects and drew those bitches".format(len(annotator.getBoxLocationsDrawn())))
            # print(annotator.getBoxLocationsDrawn())
            annotator.emptyBoxesDrawn()
            im0 = cv2.putText(im0, 'Barcodes: {}'.format(bcCt), (10,15), 0, 
                   1, colors(0,True), 1, cv2.LINE_AA)
            im0 = cv2.putText(im0, 'Strips Seen: {}'.format(stripCt), (10,45), 0, 
                   1, colors(1,True), 1, cv2.LINE_AA)
            im0 = cv2.putText(im0, 'Test Lines Seen: {}'.format(lineCt), (10,75), 0, 
                   1, colors(2,True), 1, cv2.LINE_AA)
   
            cv2.imshow(DISPLAY_WINDOW_NAME, im0)
                
            t = 1 if not type(cap) == None else 0
                
            if(cv2.waitKey(t) == ord('q')):
                done = True
                cv2.destroyAllWindows()
                quit()

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)
