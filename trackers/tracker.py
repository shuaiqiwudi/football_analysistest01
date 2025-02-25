from ultralytics import YOLO
import supervision as sv# 导入 superversion 库
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox,get_bbox_width,get_foot_position


class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()# 创建一个字节级别的跟踪器

    #添加object的位置postion到track中
    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    # 如果是球，就获取球的中心点坐标
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    # 否则获取人脚下的中心点坐标
                    else:
                        position = get_foot_position(bbox)
                    #在tracks中添加物体目标的position
                    tracks[object][frame_num][track_id]['position'] = position


    def interpolate_ball_positions(self,ball_positions):
        # 遍历每一帧的球的位置 如果球的位置为空 就用前一帧的位置进行插值
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    # 用yolo模型预测视频帧 返回检测到的物体
    def detect_frames(self,frames):
        batch_size = 20
        detections = []
        # 逐批次检测视频帧
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections+=detections_batch
        return detections

    #
    def get_object_tracks(self, frames,read_from_stub=False,stub_path=None):

        #如果已经加载过了 就从文件中读取轨迹
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            #print("track_output@@@",tracks)
            return tracks

        detections = self.detect_frames(frames)

        # 追踪的轨迹为一个dictionary
        tracks = {
            "players":[],
            "referees":[],
            "ball":[],
        }

        for frame_num,detection in enumerate(detections):
            cls_names = detection.names  # 获取检测到的物体类别
            cls_names_inv = {v:k for k,v in cls_names.items()}  # 将类别名称和类别id进行反转 方便直接观察
            print("cls_names:",cls_names)

            # Convert to supervision detection format 将yolo检测到的物体转化为supervision的检测格式
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #将门将的类别编号转化为球员的object
            for object_id ,class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_id] = cls_names_inv["player"]
                
            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            # Initialize tracks
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # 遍历每一帧的检测结果
            for frame_detection in detection_with_tracks:
                # 获取检测到的物体的边框bounding box
                bbox = frame_detection[0].tolist()
                # 获取检测到的物体的类别id
                cls_id = frame_detection[3]
                # 获取检测到的物体的轨迹id
                track_id = frame_detection[4]

                # 将检测到的物体的信息存储到tracks中
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}# frame_num表示第几帧

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            # 获取球的轨迹
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

            print("detection_with_tracks:",detection_with_tracks)

        # 保存轨迹到文件
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks

    # 画椭圆的方法
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])# 获取bbox的y2坐标 bounding box的格式是[x1,y1,x2,y2]
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # 使用opencv的ellipse画椭圆
        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),# 椭圆的一个最短直径 一个最长直径
            angle=0.0,
            startAngle=-45,# 椭圆从-45度开始画 
            endAngle=235,# 椭圆画到235度结束
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # 画椭圆上面显示track_id那个矩形 和对应的track_id
        rectangle_width = 40
        rectangle_height = 20
        # 设置矩形的左上角和右下角的xy坐标
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        #如果track_id不为空 就在椭圆上面显示track_id
        if track_id is not None:
            cv2.rectangle(frame,
                (int(x1_rect),int(y1_rect)),
                (int(x2_rect),int(y2_rect)),
                color,
                cv2.FILLED)
            # 在矩形上面显示track_id
            x1_text = x1_rect+12
            # 如果track_id大于99 就往左移动10个像素
            if track_id > 99:
                x1_text -=10
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
        return frame # 返回画好椭圆的帧
    
    # 画三角形的方法
    def draw_traingle(self,frame,bbox,color,):
        # 用bbox获取左上角的y坐标
        y= int(bbox[1]) 
        x,_ = get_center_of_bbox(bbox)
        # 三角形的三个点 x,y是倒三角形下面的点 形成数组的形式
        traingle_ponits = np.array([[x,y],[x-10,y-20],[x+10,y-20]])
        # 画三角形 cv2.FILLED表示填充坐标区域
        cv2.drawContours(frame,[traingle_ponits],0,color,cv2.FILLED)
        cv2.drawContours(frame,[traingle_ponits],0,(0,0,0),2)

        return frame
    
    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        #画控球率的播放区域
        overlay = frame.copy()
        # 放在最右下角的位置
        cv2.rectangle(overlay,(1350,850),(1900,970),(255,255,255),-1)
        # 透明度
        alpha = 0.4
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
        
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        #获取每一帧哪个队伍拥有球权 计算占据的百分比 
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        return frame
    # 
    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            #开始处理track 在其脚下生成圆圈
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # 画球员脚下的椭圆
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)# 画椭圆

                # 如果球员的has_ball为true时 在这个球员的头上画红色三角形
                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame,player["bbox"],(0,0,255))

            
            # 画裁判脚下的椭圆
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255),track_id)
            
            # 画球的三角形
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

            # 调用画控球率的方法
            frame = self.draw_team_ball_control(frame,frame_num,team_ball_control)

            output_video_frames.append(frame)
        return output_video_frames