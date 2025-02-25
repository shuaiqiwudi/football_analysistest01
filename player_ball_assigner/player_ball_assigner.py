import sys
sys.path.append('../')
from utils import get_center_of_bbox,measure_distance

class PlayerBallAssigner:
    def __init__(self):
        # 设置球员和球之间的最大距离 超出这个距离球将被认为是没有球员控制
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self, players, ball_bbox):
        #获取球的中心位置
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 1000000
        assigned_player = -1
        
        for player_id, player in players.items():
            player_bbox = player['bbox']
            # 获取球和球员任意一个foot之间的最小distance
            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player