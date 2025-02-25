from sklearn.cluster import KMeans
class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        # 根据track_id记录每个球员的队伍，我们可以将其存储在这个字典，就不需要再用kmeans算法来判断了
        self.player_teams_dict={}


    # 做和color_assignement.ipynb类似的事情
    def get_clustering_model(self, image):
        #将图像转化为2d数组
        image_2d = image.reshape(-1, 3)
        # 使用kmeans算法进行聚类 分为两类
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)
        return kmeans
    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]#裁剪出bbox区域
        top_half_image = image[:int(image.shape[0]/2)]
        kmeans = self.get_clustering_model(top_half_image)

        # 获取每个像素点的聚类标签
        labels = kmeans.labels_
        
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # get the player cluster 
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    #获取两个队伍的颜色
    def assign_team_color(self, frame, player_decetions):    
        player_colors = []
        #获取了第一帧的所有球员的RGB颜色添加到player_colors[]中
        for _,player_detection in player_decetions.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        #print("player_colors",player_colors)

        #再使用kmeans++算法对player_colors[]添加的RGB颜色分为两类
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        #将获取到的两个队伍的颜色存储到team_colors[]中
        self.team_colors[1]=kmeans.cluster_centers_[0]
        self.team_colors[2]=kmeans.cluster_centers_[1] 


    # 分配不同颜色的球员到各自的队伍
    def get_player_team(self,frame,player_bbox,player_id):
        # 如果已经分配过队伍，直接返回
        if player_id in self.player_teams_dict:
            return self.player_teams_dict[player_id]

        # 获取球员的颜色
        player_color = self.get_player_color(frame,player_bbox)
        # 不论球员是什么颜色，都用kmeans算法来判断球员属于哪个队伍
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1

        if player_id == 107:
            team_id=1
        self.player_teams_dict[player_id] = team_id

        return team_id 