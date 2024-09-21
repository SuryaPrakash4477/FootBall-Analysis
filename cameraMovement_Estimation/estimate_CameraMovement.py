import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measureDistance, measureXYDistance

class EstimateCameraMovement():
    def __init__(self, frame):
        self.minDistance = 5 #little camera movement

        self.lkParameter = dict(
            winSize  = (15, 15),
            maxLevel = 2, #to downscale the image to getg larger features
            #stopping criteria
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,0.03) #10 times loop #quality score)
        )
    
        first_framegrayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #top and bottom features
        maskFeatures = np.zeros_like(first_framegrayscale)

        #it takes the top and bottom banner
        maskFeatures[:, 0:20] = 1  #first 20 rows of pixel
        maskFeatures[:, 900:1050] = 1  #last pixel

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = maskFeatures
        )

    def add_adjust_PositiosToTracks(self, tracks, cameraMovementPerFrame):
        #adjust the positions to the tracks according to the camera movement
        for object, objectTracks in tracks.items():
            for frame_num, track in enumerate(objectTracks):
                for trackID, trackInfo in track.items():
                    position = trackInfo['position']
                    camera_Movement  = cameraMovementPerFrame[frame_num] 
                    adjustedPosition = (position[0]-camera_Movement[0], position[1]-camera_Movement[1])
                    tracks[object][frame_num][trackID]['adjustedPosition'] = adjustedPosition

    def getCameraMovement(self, frames, read_from_stub=False, stubs_path=None):
        # Read stubs file if it exists
        if read_from_stub and stubs_path is not None and os.path.exists(stubs_path):
            with open(stubs_path, 'rb') as f:
                return pickle.load(f)

        # Calculate the camera movement per frame
        cameraMovement = [[0, 0]] * len(frames)

        # Convert the image into grey image to extract features
        oldGrey = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        oldFeatures = cv2.goodFeaturesToTrack(oldGrey, **self.features)
        if oldFeatures is None:
            return cameraMovement  # No features detected in the first frame

        oldFeatures = np.float32(oldFeatures).reshape(-1, 1, 2)

        for frame_num in range(1, len(frames)):
            # Convert the image into grey image to extract features
            newGrey = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            newFeatures, status, _ = cv2.calcOpticalFlowPyrLK(oldGrey, newGrey, oldFeatures, None, **self.lkParameter)

            # Ensure newFeatures and status are not None
            if newFeatures is None or status is None:
                continue

            newFeatures = np.float32(newFeatures).reshape(-1, 1, 2)
            status = status.reshape(-1)

            # Measure distance between oldFeatures and newFeatures
            maxDistance = 0
            cameraMovement_x, cameraMovement_y = 0, 0

            for i, (new, old) in enumerate(zip(newFeatures, oldFeatures)):
                if status[i] == 1:  # Only consider good points
                    newFeatures_point = new.ravel()
                    oldFeatures_point = old.ravel()

                    distance = measureDistance(newFeatures_point, oldFeatures_point)

                    if distance > maxDistance:
                        maxDistance = distance
                        cameraMovement_x, cameraMovement_y = measureXYDistance(oldFeatures_point, newFeatures_point)
            
            if maxDistance > self.minDistance:
                cameraMovement[frame_num] = [cameraMovement_x, cameraMovement_y]
                oldFeatures = cv2.goodFeaturesToTrack(newGrey, **self.features)
                if oldFeatures is not None:
                    oldFeatures = np.float32(oldFeatures).reshape(-1, 1, 2)

            oldGrey = newGrey.copy()

        if stubs_path is not None:
            with open(stubs_path, 'wb') as f:
                pickle.dump(cameraMovement, f)

        return cameraMovement

        

    def draw_cameraMovement(self, frames, cameraMovementPerFrame):
        outputFrames = []

        for frame_num, frame in enumerate(frames):
            # frame = frame.copy() #in order to not contaminate inputed to the function

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), cv2.FILLED) # you can use -1 or cv2.FILLED for filling
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            xMovement, yMovement = cameraMovementPerFrame[frame_num]

            frame = cv2.putText(frame, f"Camera Movement X:{xMovement:.2f}",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3) #which is going to put the camera movement of x and y
            frame = cv2.putText(frame, f"Camera Movement Y:{yMovement:.2f}",(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            outputFrames.append(frame)

        return outputFrames
