from ultralytics import YOLO
import supervision as sv
import pickle
import os 
import numpy as np 
import pandas as pd
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_bbox_height

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_position(self, ball_positions):
        ball_positions = [ x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate missing values for ball positions
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    
    def detect_frames(self, frames):
        batch_size = 20 # number of frames to process at once 
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_detections = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += batch_detections

        return detections
    
    def get_object_track(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and os.path.exists(stub_path) and stub_path is not None:
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert ultralytics detection to supervision detection
            detection_sv = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player object
            for idx_object, class_id in enumerate(detection_sv.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_sv.class_id[idx_object] = cls_names_inv["player"]

            # Track objects
            detection_with_track = self.tracker.update_with_detections(detection_sv)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_track:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            # for ball detection with out tracking
            for frame_detection in detection_sv:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks
    

    def draw_ellipse(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # draw ellipse
        cv2.ellipse(
            frame,
            center= (x_center, y2),
            axes= (int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
            )
        
        # draw rectangle
        rect_width , rect_hight = int(width*1.5), int(0.8*width)
        x1_rect, x2_rect  = int(x_center - rect_width//2), int(x_center + rect_width//2)
        y1_rect, y2_rect = int(y2 - rect_hight//2) + 15, int(y2 + rect_hight//2) + 15
        cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)

        # put text
        if track_id is not None:
            cv2.putText(frame, str(track_id), (x_center - 10, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, (rect_width/100) , (0, 0, 0), 2)
        return frame

        
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        width, height = get_bbox_width(bbox), get_bbox_height(bbox)
        # triangle_points = np.array([
        #     [x,y],
        #     [x+int(width), y-int(height*1.5)],
        #     [x-int(width), y-int(height*1.5)]
        # ])
        triangle_points = np.array([
            [x,y],
            [x+10, y-20],
            [x-10, y-20]
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame
    
    def draw_annotations(self, frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # draw annotations on the frame
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))
            
            output_video_frames.append(frame)
        return output_video_frames