from sklearn.cluster import KMeans
import numpy as np
class TeamAssigner:
    def __init__(self, method='foreground'):
        self.team_colors = {0: [0, 0, 255]}
        self.player_team_dict = {}
        self.method = method

    def get_clustring_model(self, image, n_clusters=2):
        image_2d = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
        kmeans.fit(image_2d)

        return kmeans
    
    def get_player_color(self, frame, bbox):
        if self.method == 'foreground':
            image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

            top_half_image = image[0:image.shape[0]//2, 0:image.shape[1]]

            # get the clustring model
            if top_half_image.size != 0:
                kmean = self.get_clustring_model(top_half_image)

                # get the cluster labels for each pixel
                labels = kmean.labels_

                # reshape the labels to the image shape
                clustred_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

                # get player cluster 
                corner_cluster = [clustred_image[0,0], clustred_image[0,-1], clustred_image[-1,0], clustred_image[-1,-1]]
                non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
                player_cluster = 1 - non_player_cluster

                player_color = kmean.cluster_centers_[player_cluster]
                return player_color
            else:
                player_color = None
                return player_color
            
        elif self.method == 'center_box':
            image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            image_center_box = image[image.shape[0]//4:(2*image.shape[0])//4, image.shape[1]//3:(3*image.shape[1])//5]

            if image_center_box.size != 0:
                kmean = self.get_clustring_model(image_center_box, n_clusters=5)

                # get the cluster labels for each pixel
                labels = kmean.labels_

                # get player cluster 
                player_color = kmean.cluster_centers_[np.argmax([sum(labels==i) for i in range(5)])]
                # player_color = kmean.cluster_centers_[np.argsort([sum(labels==i) for i in range(5)])]
                # take avrage of the two highest clusters
                # player_color = np.mean(player_color[:2], axis=0)
                return player_color
            else:
                player_color = None
                return player_color

    def assign_team_color(self, frame, players_detections):
        
        players_colors = []
        for _, player_detection in players_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            players_colors.append(player_color)
        
        # cluster two teams colors
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(players_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def assign_players_to_teams(self, frame, frame_num, player_bbox, player_id):

        # predict the team of the player
        def predict_team():
            player_color = self.get_player_color(frame, player_bbox)

            if player_color is None:
                return 0
            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] 

            # to make teams id 1 and 2 instead of 0 and 1
            team_id += 1

            self.player_team_dict[player_id] = team_id

            return team_id
        
        # # predict the team every 7 frames
        # if frame_num % 7 == 0:
        #     return predict_team()
        
        # # if the player is not detected in the frame
        # else:
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        else:
            return predict_team()
            