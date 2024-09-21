from ultralytics import YOLO
import supervision as sv    #to track after prediction
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils import get_centerOfbbox, getbbox_width, get_FootPosition

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_PositionToTracks(self, tracks):
        #make player position robust to the camera movement sp for that you have to get the player postion
        #  and the position to the tracks first and then subtract the camera movement from the player psotions
        for object, objectTracks in tracks.items():
            for frame_num, track in enumerate(objectTracks):
                #get the player position relative to the bbox
                for trackID, trackInfo in track.items():
                    bbox = trackInfo['bbox']
                    if object == 'ball':
                        #position is center of it
                        position = get_centerOfbbox(bbox)
                    else:
                        #want the foot position which is center of the bbox but the center of the terms of 
                        # X but the bottom in terms of Y
                        position = get_FootPosition(bbox)
                    tracks[object][frame_num][trackID]['position'] = position
        

    def interpolate_BallPosition(self, ball_positions):
        #function to draw trianagle in some frame where it doesn't able to track it
        ##interpolation is missing in the missing value and pandas is going to help you
        #to interpolate the missing values because it already has a function for it.

        #convert ball position to pandas dataframe
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions] #[] going to be interpolted by pandas dataframe
        df_ballPositions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        #interpolate the missing values
        df_ballPositions = df_ballPositions.interpolate() #it fills the 99% missing value

        #if the edge case is first frame than thn it is not going to interpolate
        # you can do by replicating it to nearest detection
        df_ballPositions = df_ballPositions.bfill() #it fills only 2-3 missing value

        #save it back into ball_positions
        ball_positions = [{1: {"bbox": x}}for x in df_ballPositions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        #add batch size to avoid memory issue
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            detections += detections_batch
        return detections

    def get_objectTracker(self, frames, read_from_stub = False, stub_path = None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        #Detect first 
        detections = self.detect_frames(frames)

        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_name = detection.names
            cls_name_inv = {v:k for k,v in cls_name.items()}
            print(cls_name)

            #Converting the detection into supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Converting the goalkeeper as a normal player
            for object_idex, class_ID in enumerate(detection_supervision.class_id):
                if cls_name[class_ID] =="goalkeeper":
                    detection_supervision.class_id[object_idex] = cls_name_inv["player"]

            #Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})  #key= tracker_id and value= bounding box
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_name_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_name_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_name_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            # Save the tracks to a file
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3])
        xcenter, _ = get_centerOfbbox(bbox)
        width = getbbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(xcenter, y2),
            axes=(int(width), int(0.50*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=4,
            lineType=cv2.LINE_4
        ) 

        rect_width = 40
        rect_height = 20
        x1rect = xcenter - rect_width//2
        x2rect = xcenter + rect_width//2
        y1rect = (y2 - rect_height//2) + 15
        y2rect = (y2 + rect_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1rect), int(y1rect)),
                          (int(x2rect), int(y2rect)),
                          color,
                          cv2.FILLED)
            

            x1text = x1rect + 12
            if track_id > 99:
                x1text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1text),int(y1rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                4
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_centerOfbbox(bbox)

        trianglePoints = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ])

        cv2.drawContours(frame, [trianglePoints], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [trianglePoints], 0, (0, 0, 0), 2) #for border

        return frame

    def draw_team_ballControl(self,frame, frame_num, team_ballControl):
        # Draw semi-transparent rectanle team on the bottom of frame
        overlay = frame.copy() #help you with transparency
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1- alpha, 0, frame) #semi-transparent rectangle

        #Calculate the % of holding ball by each team and put the into the rectangle
        team_ballControl_tillframe = team_ballControl[:frame_num+1]
        team_1_num_frames = team_ballControl_tillframe[team_ballControl_tillframe == 1].shape[0]
        team_2_num_frames = team_ballControl_tillframe[team_ballControl_tillframe == 2].shape[0]
        
        #make statistics for each team
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball control: {team_1*100: .2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(frame, f"Team 2 Ball control: {team_2*100: .2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)

        return frame


    def draw_annotations(self, video_frames, tracks, team_ballControl):
        output_video_frame = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            #draw circle on players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("ball_acquired", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            #draw circle beneath referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            #draw triangle on ball

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            #draw team ball control
            frame = self.draw_team_ballControl(frame,frame_num, team_ballControl)

            output_video_frame.append(frame)

        return output_video_frame