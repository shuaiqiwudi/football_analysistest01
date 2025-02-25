# 获取bbox的中心点坐标
def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# 获取bbox的宽度
def get_bbox_width(bbox):
    return bbox[2] - bbox[0] # x2 - x1

# 计算两个点之间的距离
def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

# x坐标的取中间值，y坐标取最底部的值
def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)