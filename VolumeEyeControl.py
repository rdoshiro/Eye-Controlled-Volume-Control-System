import cv2              #importing library
import time
import numpy as np
import Module as md
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 480, 480   #width and height of camera





#turning on webcam
cap = cv2.VideoCapture(0)
cap.set(2, wCam)
cap.set(2, hCam)
pTime = 0

detector = md.FaceDetection(minDetectionCon=0.7) #creating object for facedection class

#audio control
devices = AudioUtilities.GetSpeakers()         #intialization
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minRange = volRange[0]       #min volume (volume ranges from -144(lowest),0(highest)) using list[0] for first value in range
maxRange = volRange[1]       #max volume (list[1] represent send value in the list)

while True:
    sucess, img = cap.read()
    img = detector.findMeshFaces(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        #print(lmList[4], lmList[12], lmList[19],lmList[29] )


        x1, y1 = lmList[5][1], lmList[5][2]         #getting x and y values for each of the landmarks
        x2, y2 = lmList[13][1], lmList[13][2]
        x3, y3 = lmList[21][1], lmList[21][2]
        x4, y4 = lmList[30][1], lmList[30][2]


        cv2.circle(img, (x1, y1),1, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2),1, (255, 0, 0), cv2.FILLED) #drawing cirle for the point
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1) #drawing line between the points
        cv2.circle(img, (x3, y3),1, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x4, y4),1, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 1)


        l1 = math.hypot(x2 - x1, y2 - y1)
        l2=math.hypot(x4 - x3, y4 - y3)
        print(l1,l2)

        if 8 < l1 < 12 and 8 < l2 < 12:
            avg_vol = (l1+l2)/2
            vol = np.interp(avg_vol, [12], [maxRange])
            volume.SetMasterVolumeLevel(vol, None)

        elif l1 < 7 and l2 < 7:
            avg_vol = (l1+l2)/2
            vol = np.interp(avg_vol,[7], [minRange])
            volume.SetMasterVolumeLevel(vol, None)




        ''''
        vol = np.interp(l1,[7,12], [minRange, maxRange])
        vol2 = np.interp(l2,[7,12], [minRange, maxRange])

        print(vol, vol2)
        volume.SetMasterVolumeLevel(vol, None)'''


        if l1 > 7:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        if l2 > 7:
            cv2.line(img, (x3, y3), (x4, y4), (0, 0, 255), 1)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}',(20,70), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2) #printing frame rate
    cv2.imshow('Img', img)
    cv2.waitKey(1)