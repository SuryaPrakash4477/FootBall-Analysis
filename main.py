from utils import read_video, save_video
from tracker import Tracker
import cv2
from team_assigner import AssignTeam
from Ball_AssignedToPlayer import BallAssignedToPlayer
import numpy as np

#to calculate the camera movement and then also draw it
from cameraMovement_Estimation import EstimateCameraMovement
from viewTransformer import ViewTransformer
from speedAndDistanceEstimation import SpeedAndDistanceEstimation

def main():
    #read video
    video_frame = read_video(rf'input_video\08fd33_4.mp4')

    #Initialize tracker and then do prediction 
    tracker = Tracker(rf'models\best.pt')

    tracks = tracker.get_objectTracker(video_frame, read_from_stub= True, 
                                       stub_path="stubs/tracks_file.pkl")
    
    #get object positions
    tracker.add_PositionToTracks(tracks)

    #Estimate the camera movement 
    estimate_cameraMovement = EstimateCameraMovement(video_frame[0])
    camera_MovementPerFrame = estimate_cameraMovement.getCameraMovement(video_frame,
                                                                        read_from_stub=True,
                                                                        stubs_path='stubs/camera_movementStub.pkl')
    
    estimate_cameraMovement.add_adjust_PositiosToTracks(tracks, camera_MovementPerFrame)

    #View Transformer
    viewTransformer = ViewTransformer()
    viewTransformer.add_TransformedPostion_ToTracks(tracks)

    #interpolate ball positions
    tracks["ball"] = tracker.interpolate_BallPosition(tracks["ball"])

    # #save cropped image of a player for color analysis
    # for track_id, player in tracks['player'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frame[0]

    #     #crop bbox from frame
    #     cropped_frame = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     #save the cropped image

    #     cv2.imwrite(f'output_video/cropped_frame.jpg', cropped_frame)
    #     break

    #speed and Distance Estimation
    speedAndDistanceEstimation = SpeedAndDistanceEstimation()
    speedAndDistanceEstimation.add_SpeedAndDistanceToTracks(tracks)

    #Assign player team
    assignteam = AssignTeam()
    assignteam.assignTeamColor(video_frame[0], 
                               tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for playerID, track in player_track.items():
            team = assignteam.assign_playertoTeam(video_frame[frame_num],
                                                  track['bbox'],
                                                  playerID)
            
            tracks['players'][frame_num][playerID]['team'] = team
            tracks['players'][frame_num][playerID]['team_color'] = assignteam.teamColors[team]

    #Assign ball aquisition
    playerAssigner = BallAssignedToPlayer()
    team_ballControl = [] #it has frame number and the frame number is going to be assigned a team
    for frame_num, player_track in enumerate(tracks['players']):
        ballbbox = tracks['ball'][frame_num][1]['bbox']
        assignedPlayer = playerAssigner.assign_BallToPlayer(player_track, ballbbox)

        if assignedPlayer != -1:
            tracks['players'][frame_num][assignedPlayer]['ball_acquired'] = True #ball_acquired is new parameter into the dictionary
            team_ballControl.append(tracks['players'][frame_num][assignedPlayer]['team'])
        else:
            team_ballControl.append(team_ballControl[-1]) #last person who has the ball
    #so the list should be numpy if not you have to make it        
    team_ballControl = np.array(team_ballControl)

    #Draw Output
    ##Draw Object tracks
    output_videoframes = tracker.draw_annotations(video_frame, tracks, team_ballControl)

    #Draw camera movement
    output_videoframes = estimate_cameraMovement.draw_cameraMovement(output_videoframes, camera_MovementPerFrame)

    #draw speed and distance
    speedAndDistanceEstimation.draw_SpeedAndDistance(output_videoframes, tracks)

    #save video
    save_video(output_videoframes, 'output_video/output_video.avi')
    
    # plt.imshow(output_videoframes)
    # plt.show()
if __name__ == "__main__":
    main()