from trackers import Tracker
from utils import read_video, save_video
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
import numpy as np
import cv2
from view_transformer import ViewTransformer

def main():
    #read video
    video_frames = read_video('input_video/08fd33_4.mp4')


    # initailize tracker
    tracker = Tracker('models/best.pt')


    #get object tracks 将视频转变为supervision形式，获取物体轨迹
    tracks = tracker.get_object_tracks(video_frames,
                                    read_from_stub=True,
                                    stub_path='stubs/track_stubs.pkl')

    #获取物体目标位置
    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,read_from_stub=True,stub_path='stubs/camera_movement_stub.pk1')

    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    

    # 视角转换
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # interpolate ball positions为球在一些帧丢失的position进行插值
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])


    # 分派球员到队伍
    team_assigner = TeamAssigner()
    # 首先给第一帧 识别两个队伍的颜色
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    # 给每个球员分配队伍
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # 分派球给队员
    player_assigner =PlayerBallAssigner()

    # 用于存储控球的队伍 ：一维数组 存储每一帧的球权属于哪个队伍
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            # 在tracks中获取球员的队伍添加到控球数组
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            
        else:
            #如果球在被传或者其他情况的过程中 那么这个球的所有权属于上一个球员
            team_ball_control.append(team_ball_control[-1])

    team_ball_control= np.array(team_ball_control)
            
    

    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

    # 画显示摄像机的运动数据的区域
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)



    #save video
    save_video(output_video_frames,'output_video/output_video.avi')
if __name__ == '__main__':
    main()