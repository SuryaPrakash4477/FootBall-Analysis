from sklearn.cluster import KMeans

class AssignTeam:
    def __init__(self):
        self.teamColors = {}
        self.player_teamDict = {}

    def get_clusteringModel(self, image):
        #reshape the image into the 2d array
        image_2d  = image.reshape(-1, 3)

        #perform K-means with 2 cluster
        ##we want least number of iteration so "k-means++" help you to get better clusters faster 
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans 
    
    def get_playerColor(self, frame, bbox):
        Frame = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_halfofFrame = Frame[0:int(Frame.shape[0]/2), :]

        #get clustering model
        kmeans = self.get_clusteringModel(top_halfofFrame)

        #get cluster labels for each pixel
        labels = kmeans.labels_

        #reshape the labels to the image shape
        clusteredFrame = labels.reshape(top_halfofFrame.shape[0], top_halfofFrame.shape[1])

        #get the class for the corner like if the class is 0 0 0  tghen class 0 is the background and class 1 is the foreground
        corner_clusters = [clusteredFrame[0, 0], clusteredFrame[0, -1], clusteredFrame[-1, 0], clusteredFrame[-1, -1]]
        non_palyercluster = max(set(corner_clusters), key=corner_clusters.count) #non_playercluster is the number which appeared most on those coerner

        playercluster = 1 - non_palyercluster

        playerColor = kmeans.cluster_centers_[playercluster]

        return playerColor

    def assignTeamColor(self, frame, player_detections):
        #Assign team color to each player in frame
        playerColors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_playerColor(frame, bbox)
            playerColors.append(player_color)


        #divide the colors into white and green
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(playerColors)

        self.kmeans = kmeans

        self.teamColors[1] = kmeans.cluster_centers_[0]
        self.teamColors[2] = kmeans.cluster_centers_[1]

        #after this you have to match the player t-shirt color with team color


    def assign_playertoTeam(self, frame, playerbbox, playerID):
        if playerID in self.player_teamDict:
            return self.player_teamDict[playerID]

        playerColor = self.get_playerColor(frame, playerbbox)

        team_id = self.kmeans.predict(playerColor.reshape(1, -1))[0] #because the team id going to be 0 or 1
        team_id +=1

        if playerID == 81:
            team_id = 2
        elif playerID == 188:
            team_id = 1

        self.player_teamDict[playerID] = team_id

        return team_id