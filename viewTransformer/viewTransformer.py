import numpy as np
import cv2

class ViewTransformer:
    #To calculate the distance covered by player in real time
    def __init__(self):
        courtWidth = 68
        courtLength = 23.32

        #trapezoid
        self.pixelVerticies = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ])
        #rectangle this is what you get after the transformation of trapezoid
        self.targetVerticies = np.array([
            [0, courtWidth],
            [0, 0],
            [courtLength, 0],
            [courtLength, courtWidth],
        ])

        self.pixelVerticies = self.pixelVerticies.astype(np.float32)
        self.targetVerticies = self.targetVerticies.astype(np.float32)

        #you have model that will switch between pixel vertices to real world vertices
        self.perspectiveTransformer = cv2.getPerspectiveTransform(self.pixelVerticies, self.targetVerticies)

        #afterwards you have to transform the adjusted points to the points after the 
        #transformation of the perspective_transformation

    def transformPoint(self, point):
        p = (int(point[0]), int(point[1]))

        #check whether the points is inside the trapezoid or not
        isInside = cv2.pointPolygonTest(self.pixelVerticies, p, False) >= 0
        if not isInside:
            return None
        
        reshapedPoint = point.reshape(-1, 1, 2).astype(np.float32) #reshap the poin so that it is readable by perspective transformer
        transformPoint = cv2.perspectiveTransform(reshapedPoint, self.perspectiveTransformer)

        return transformPoint.reshape(-1, 2)

    def add_TransformedPostion_ToTracks(self, tracks):
        #transform the points to the real world points
        for object, objectTracks in tracks.items():
            for frame_num, track in enumerate(objectTracks):
                for trackID, trackInfo in track.items():
                    position = trackInfo['adjustedPosition']
                    position = np.array(position)
                    transformedPosition = self.transformPoint(position)
                    if transformedPosition is not None:
                        transformedPosition = transformedPosition.squeeze().tolist()
                    tracks[object][frame_num][trackID]['transformedPosition'] = transformedPosition