import requests
import json

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

