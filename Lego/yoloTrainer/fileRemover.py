import os
import time

path = "."

xmlNames = []
imgNames = []
print(len(os.listdir(path)))
i = 0
for file in os.listdir(path):
    i+=1
    try:
        name = file.split(".")[0]
        type = file.split(".")[1]
        if ( "txt" in type):
            xmlNames.append(name)
        elif "py" not in type:
            imgNames.append(file)
    except:
        print("Got a folder")
print(len(imgNames))
print(len(xmlNames))

toDel = 0
for name in imgNames:
    checkForName = name.split(".")[0]
    if checkForName not in xmlNames:
        toDel +=1
        os.remove(name)
   
print(len(os.listdir(path)))     

print(toDel)