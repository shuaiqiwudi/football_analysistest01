import cv2
#读取视频，保存视频
def read_video(video_path):
    #读取视频
    cap = cv2.VideoCapture(video_path)
    #获取视频帧率
    frames = []
    #读取视频帧
    while True:
        #ret:是否读取到视频帧
        ret,frame = cap.read()
        if not ret:
            break
        # 将读取到的视频帧保存到frames中
        frames.append(frame)
    return frames

def save_video(ouput_video_frames,output_video_path):
    #保存视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path,fourcc, 20.0, (ouput_video_frames[0].shape[1],ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()

    