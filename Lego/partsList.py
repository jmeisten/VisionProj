import requests
import json
import cv2
import tensorflow as tf

urlBase = "https://rebrickable.com/api/v3/lego/"

# TODO: Replace this with read from txt file
apiKey = "3c72de2d0ef5c79d6331ec6ca12cf0fe"

def getPartsForKit(kitNumber):
    global urlBase
    global apiKey
    retVal = []
    url = urlBase + "sets/"+ kitNumber + "/parts/"
    print(url)
    response = requests.get(url, params={'key': apiKey})
    if (response.ok):
        results = response.json()["results"]
        # print(response.json())
        
        for jobj in results:
            retVal.append(jobj["part"]["part_num"])
            # print(jobj["inv_part_id"])
        
    return retVal
    
def getColorsForPart(partNum):
    global urlBase
    global apiKey
    retVal = []
    url = urlBase + "parts/"+ partNum + "/colors/"
    print(url)
    response = requests.get(url, params={'key': apiKey})

    if (response.ok):
        results = response.json()["results"]
        for jobj in results:
            retVal.append(jobj["color_id"])
        
    return retVal

def getKitsForPart(partNum):
    global urlBase
    global apiKey
    retVal = []
    colors = getColorsForPart(partNum=partNum)
    for i in range(len(colors)):
        
        url = urlBase + "parts/"+ partNum + "/colors/"+ str(colors[i]) +"/sets/"
        print(url)
        response = requests.get(url, params={'key': apiKey})

        if (response.ok):
            results = response.json()["results"]
            for jobj in results:
                name = jobj["name"].split('-')[0]
                val = {jobj["set_num"],name}
                retVal.append(val)
                # print(jobj["inv_part_id"])
        
    return retVal

def makeLDRAWFile(kit="31072-1"):
    LDRAW_File= open("LDRAW_file.ldr","w")
    parts = getPartsForKit(kit)
    for piece in parts:
        line = " 1 4 0 0 0 1 0 0 0 1 0 0 0 1 " + str(piece)+'.dat'
        LDRAW_File.write(line)
        LDRAW_File.write("\n")
    LDRAW_File.close()


cap = cv2.VideoCapture(0)
interpreter = tf.lite.Interpreter(model_path="./yolo/models/31072-1.tflite")
interpreter.allocate_tensors()
inputDetails = interpreter.get_input_details()
outputDetails = interpreter.get_output_details()

while True:
    ret, img = cap.read()
    
    if not ret:
        continue
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgRGB = imgRGB.resize((640,640))
    imgRGB = imgRGB.reshape(1,imgRGB.shape[0],imgRGB.shape[1],3)
    print(inputDetails)
    interpreter.set_tensor(float(inputDetails[0]['index']),imgRGB)
    interpreter.invoke()
    detectedBoxes = interpreter.get_tensor(outputDetails[0]['index'])[0]
    detectedClasses = interpreter.get_tensor(outputDetails[0]['index'])[0]
    detectedCT = interpreter.get_tensor(outputDetails[0]['index'])[0]
    
    print("\n{} \n{} \n{}\n".format(detectedBoxes,detectedClasses,detectedCT))