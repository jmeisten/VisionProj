
import os
import torch
from IPython.display import Image
import os
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False


# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    infoDict = {}
    infoDict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            infoDict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            infoDict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            infoDict['bboxes'].append(bbox)
    
    return infoDict


def convert_to_yolo(infoDict):
    printBuffer = []
    classIDs = []
    #loop over every box in the PASCAL VOC
    for b in infoDict["bboxes"]:

        classID = b["class"]
        classIDs.append(classID)
        # box coords to yolo v5 format
        bCenterX = (b["xmin"] + b["xmax"]) / 2
        bCenterY = (b["ymin"] + b["ymax"]) / 2
        height = b["ymax"] + b["ymin"] 
        width = b["xmax"] + b["xmin"] 

         # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = infoDict["image_size"]  
        bCenterX /= image_w 
        bCenterY /= image_h 
        width    /= image_w 
        height   /= image_h 

        #Write the bbox details to the file 
        printBuffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(classID, bCenterX, bCenterY, width, height))
        
    # Name of the file which we have to save 
    save_file_name = os.path.join("annotations", infoDict["filename"].replace("png", "txt"))
    namesOfClasses =  "classes.txt"
    
    # Save the annotation to disk
    print("\n".join(printBuffer), file= open(save_file_name, "w"))
    if os.path.exists(namesOfClasses) == True:
        print("\n".join(classIDs), file= open(namesOfClasses, "a"))
    else:
        print("\n".join(classIDs), file= open(namesOfClasses, "w"))

    return printBuffer
    

# convert PASCAL VOC to Yolo v5 style annotatios

annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "xml"]

# some quick file clean ups
try:
    os.remove("classes.txt")
    os.remove("classesString.txt")
except Exception as e:
    print(e)

# Convert and save the annotations
for ann in tqdm(annotations):
    infoDict = extract_info_from_xml(ann)
    convert_to_yolo(infoDict)
print("Finished grabbing all annotations and converting to txt")
print("Converting classes file to text array for yaml file")

file = open('classes.txt','r')
lines = file.readlines()
classesString =""
classesArr =[]
for line in lines:
    if len(line.strip()) > 2:

        line = line.replace('"',"")
        line = line.strip()
        if not line in classesArr:
            classesArr.append(line)
            classesString += '"' + line+ '",'
print("Number of unique classes in annotations for images: {}".format(len(classesArr)))
classesString = classesString[0:-1]

try:
    f = open("classesString.txt","w")
    f.write(classesString)
    f.close()
except Exception as e:
    print(e)

print("Done")

annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]

# partion dataset

# read images and annotations
images = [os.path.join('images', x) for x in os.listdir('images') if x[-3:]=="png"]
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]

images.sort()
annotations.sort()


# Split the dataset into train-valid-test splits 
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

# Move the splits into their folders
move_files_to_folder(train_images, 'images/train')
move_files_to_folder(val_images, 'images/val/')
move_files_to_folder(test_images, 'images/test/')
move_files_to_folder(train_annotations, 'labels/train/')
move_files_to_folder(val_annotations, 'labels/val/')
move_files_to_folder(test_annotations, 'labels/test/')

# Training options

# img : size of image in a square. OG img while maintaining aspect ratio
# batch : batch size
# epochs: umber of epochs to train for
# data : Data yaml file that contains dataset info (paths)
# workers : number of cpu workers
# cfg: model architecture can use yolo5s.yaml ,5m,5l or 5x with s being the simplest and x being the most complex
# weights: weights you want to start training from
#               --weights ' ' for default
# name: various things about traing like logs and shit
# hyp : hyperparameters ?

