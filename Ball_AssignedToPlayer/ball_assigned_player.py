import sys
sys.path.append('../')
from utils import get_centerOfbbox, measureDistance

class BallAssignedToPlayer:
    def __init__(self):
        self.max_ball_playerDistance = 70 #if the distance geater then 70 then the ball will not assign to anyone

    def assign_BallToPlayer(self,players, ball_bbox):
        ballPosition = get_centerOfbbox(ball_bbox)

        minimumDistance = 99999 #get the closest player
        assignedPlayer = -1

        for playerId, player in players.items():
            playerbbox = player['bbox']

            distance_leftfoot = measureDistance((playerbbox[0], playerbbox[-1]), ballPosition)
            distance_rightfoot = measureDistance((playerbbox[2], playerbbox[-1]), ballPosition)

            distance = min(distance_leftfoot, distance_rightfoot)

            if distance < self.max_ball_playerDistance:
                if distance < minimumDistance:
                    minimumDistance = distance
                    assignedPlayer = playerId

        return assignedPlayer

