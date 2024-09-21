import cv2
import sys
sys.path.append('../')
from utils import measureDistance, get_FootPosition

class SpeedAndDistanceEstimation:
    def __init__(self):
        #current speed of the player which is going to be within certain frame number
        # speed of the player every five frame
        self.frameWindow = 5
        #frame rate which is going to be frames per second
        self.frameRate = 24

    def add_SpeedAndDistanceToTracks(self, tracks):
        totalDistance = {}

        #calculate the speed for each object
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue
            #calculate the speed of the player
            numberOfFrame = len(object_tracks)
            for frame_num in range(0, numberOfFrame, self.frameWindow):
                #define last frame because sometimes when you add frame the last frame might be =5 of less than that
                #this is just to not go out of bound
                lastFrame = min(frame_num +self.frameWindow, numberOfFrame - 1)

                for trackId, _ in object_tracks[frame_num].items():
                    if trackId not in object_tracks[lastFrame]:
                        continue
                    #to calculate the speed of the player you will need to identfy that the player should exist into the first frame
                    # of above batch or last frame of the above batch
                    startPosition = object_tracks[frame_num][trackId]["transformedPosition"]
                    endPosition = object_tracks[lastFrame][trackId]["transformedPosition"]

                    #check the start and end position of player
                    #if it is none means the player outside of the trapezoidal then you don't have to calculate the speed or distance
                    if startPosition is None or endPosition is None:
                        continue

                    #otherwise measure the distance and calculate the speed of player
                    distanceCovered = measureDistance(startPosition, endPosition)
                    timeElapsed = (lastFrame-frame_num)/self.frameRate
                    speedMetersPerSecond = distanceCovered/timeElapsed
                    speedKMPerHour = speedMetersPerSecond*3.6

                    #saving the distance
                    if object not in totalDistance: #make it in dictionary
                        totalDistance[object] = {}

                    if trackId not in totalDistance[object]:
                        totalDistance[object][trackId] = 0

                    totalDistance[object][trackId] += distanceCovered

                    #speed and distance lbel annotations
                    for frame_numBatch in range(frame_num, lastFrame):
                        if trackId not in tracks[object][frame_numBatch]:
                            continue
                        tracks[object][frame_numBatch][trackId]['speed'] = speedKMPerHour
                        tracks[object][frame_numBatch][trackId]['distance'] = totalDistance[object][trackId]

    def draw_SpeedAndDistance(self, videoframe, tracks):
        #draw speed and distance underneath the player
        outputFrame = []
        for frame_num, frame in enumerate(videoframe):
            for object, objectTracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue
                for _, trackInfo in objectTracks[frame_num].items():
                    if "speed" in trackInfo:
                        speed = trackInfo.get("speed", None)
                        distance = trackInfo.get("distance", None)
                        if speed is None or distance is None:
                            continue
                        #wrtie the speed and distance below the bounding box
                        bbox = trackInfo['bbox']
                        position = get_FootPosition(bbox)
                        position = list(position) #buffer underneath the bbox
                        position[1] +=40

                        position = tuple(map(int, position))

                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            outputFrame.append(frame)
        return outputFrame

