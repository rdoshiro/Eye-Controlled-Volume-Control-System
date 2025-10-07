import cv2                          #importing library
import mediapipe as mp
import time
import numpy as np


class FaceDetection(): #creating a class
    def __init__(self, staticMode=True, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5): #initializing parameters
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon


        self.mpDraw = mp.solutions.drawing_utils  #creating fash mesh using landmark
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon) #tracking just one face
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        # Define landmark indices for eyes
        self.LEFT_IRIS_LANDMARKS = [471, 470, 469, 472]  # Left iris landmarks
        self.RIGHT_IRIS_LANDMARKS = [475, 474, 476, 477]  # Right iris landmarks

        self.eye_landmarks_left =  [162,33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246] #landmarks
                                    #46,53,52,65,55,70,105,66,107
        self.eye_landmarks_right = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398] #landmarks
                                    #285,295,282,283,276,300,293,334,296,336

    def findMeshFaces(self, img, draw=True):        #initializing method
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #converting the image to RBG
        self.results = self.faceMesh.process(self.imgRGB)

        if self.results.multi_face_landmarks:        #drawing landmark on face
            for faceLms in self.results.multi_face_landmarks:
                #self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,self.drawSpec,self.drawSpec)
                #if draw:
                    '''for id, lm in enumerate(faceLms.landmark):         #printing id for the point, landmark x and y locations
                        #print(lm.x, lm.y, lm.z)
                        ih, iw, ic = img.shape
                        x,y = int(lm.x * iw), int(lm.y * ih)
                        #pts = np.array([(int(lm.x * iw), int(lm.y * ih))])'''
                        # Draw only the selected landmarks (eyes and nose)
                        #if id in  self.eye_landmarks_left +  self.eye_landmarks_right:
                            #cv2.circle(img, (x, y), 1,(0, 255, 50), -1)
                            #cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)

                    #cv2.circle(img, (x, y), 1, (0,255, 50), -1)  # Blue for left eye

        return img

    def findPosition(self, img, eyeno=0, draw = True):
        lmList = []
        if self.results.multi_face_landmarks:
            myeye= self.results.multi_face_landmarks[eyeno]
            ih, iw, ic = img.shape

            selected_id = self.eye_landmarks_left + self.eye_landmarks_right
            #for id, lm in enumerate(myeye.landmark):         #printing id for the point, landmark x and y locations
                #print(lm.x, lm.y, lm.z)
             #   ih, iw, ic = img.shape

            for id in selected_id:
                if id >= len(myeye.landmark):
                    continue
                lm = myeye.landmark[id]
                x,y = int(lm.x * iw), int(lm.y * ih)
                lmList.append([id,x,y])

                if draw:
                    cv2.circle(img,(x,y),1,(255,0,0),-1)
               # x,y = int(lm.x * iw), int(lm.y * ih)

               # lmList.append([id,x,y])
              #  if id in  self.eye_landmarks_left +  self.eye_landmarks_right:
             #       cv2.circle(img, (x, y), 1,(0, 255, 50), -1)


        return lmList

def main():
    cap = cv2.VideoCapture(0)  # giving webcam input
    pTime = 0
  #  mpDraw = mp.solutions.drawing_utils  #drawing lines between the points
    detector = FaceDetection() #calling class


    while True:
        success, img = cap.read()  # reading webcam
        img = detector.findMeshFaces(img)
        lmList = detector.findPosition(img)
        #if len(lmList) != 0:
         #   print(lmList[46])
        cTime = time.time()                     #frame rate (use for detect how smooth the video output is)
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}',(20,70), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2) #printing frame rate
        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()