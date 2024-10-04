from sklearn.cluster import KMeans
import numpy as np
import torch
from transformers import AutoProcessor, SiglipVisionModel
from tqdm import tqdm
import supervision as sv
import cv2 
from more_itertools import chunked
import umap

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TeamAssigner:
    def __init__(self, frames, detections, method='foreground'):
        self.team_colors = {0: [0, 0, 255]}
        self.player_team_dict = {}
        self.method = method
        if self.method == 'siglip':
            self.clusters, self.frame_player_map = Siglip().pipeline(frames, detections)


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
            
        elif self.method == 'center_box' or self.method == 'siglip':
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
        if self.method == 'siglip' or self.method == 'center_box':
            self.team_colors[1] = [0, 0, 255] # red
            self.team_colors[2] = [0, 255, 0] # green
        else:
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



    def assign_players_to_teams(self, frame, frame_num, player_bbox, player_id, player_idx):
        if self.method == 'siglip':
            def predict_team():
                embedding_index = self.frame_player_map.get((frame_num, player_idx))
                if embedding_index is not None:
                    team_id = self.clusters[embedding_index] + 1
                    self.player_team_dict[player_id] = team_id
                    return team_id
                else:
                    return 0
        else:
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
        
        if player_id not in self.player_team_dict or self.method == 'siglip':
            return predict_team()

        else:
            return self.player_team_dict[player_id]


    
class Siglip():
    def __init__(self):
        self.EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH, attn_implementation="sdpa").to(DEVICE).eval()
        self.EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH,)
        self.REDUCER = umap.UMAP(n_components=3)
        self.CLUSTERING_MODEL = KMeans(n_clusters=2)
        self.clusters = {}
        self.frame_player_map = {}

    def crop_players(self, frames, detections):
        PLAYER_ID = 2
        crops = []
        embedding_index = 0
        self.frame_player_map = {}

        for frame_idx, (frame, detection) in enumerate(zip(frames, detections)):
            # Filter detections by the class ID (e.g., PLAYER_ID)
            player_detections = detection[detection.class_id == PLAYER_ID]
            
            # Crop images for each player detected in the frame
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]

            # New: Map each player in each frame to an embedding index
            for player_idx, _ in enumerate(players_crops):
                self.frame_player_map[(frame_idx, player_idx)] = embedding_index
                embedding_index += 1
            
            # Collect all crops
            crops += players_crops

        return crops
    
    def embedding_players(self, crops, batch_size=64):

        batches = chunked(crops, batch_size)
        data = []
        for batch in tqdm(batches, desc='embedding extraction'):
            embeddings = self.embedding_(batch)
            data.append(embeddings)

        data = np.concatenate(data)
        return data
    
    @torch.no_grad()
    def embedding_(self, player):
        inputs = self.EMBEDDINGS_PROCESSOR(images=player, return_tensors="pt").to(DEVICE)
        outputs = self.EMBEDDINGS_MODEL(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        return embeddings
    
    def fit(self, data):
        projections = self.REDUCER.fit_transform(data)
        self.CLUSTERING_MODEL.fit(projections)

    def predict(self, data, is_umap=True):
        if is_umap:
            projections = self.REDUCER.transform(data)
        else:
            projections = data

        cluster_model = self.CLUSTERING_MODEL.predict(projections)
        for i in range(len(data)):
            self.clusters[i] = cluster_model[i]
        return cluster_model
    
    def pipeline(self, frames, detections):
        crops = self.crop_players(frames, detections)
        data = self.embedding_players(crops)
        self.fit(data)
        return self.predict(data), self.frame_player_map
