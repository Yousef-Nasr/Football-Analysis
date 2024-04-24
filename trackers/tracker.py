from ultralytics import YOLO
import supervision as sv
import pickle
import os 
import cv2
import utils

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

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
    

    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = bbox[3]
        center = utils.get_center_of_bbox(bbox)

        
    def draw_annotations(self, frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)