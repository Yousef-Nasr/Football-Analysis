from utils import read_video, save_video
from trackers import Tracker


def main():
    video_frames = read_video("input_videos/goal2.mp4")
    tracker = Tracker("models/best.pt")
    #tracks = tracker.get_object_track(video_frames, read_from_stub=True, stub_path='stubs/stub_tracks.pkl')
    tracks = tracker.get_object_track(video_frames, read_from_stub=False, stub_path='stubs/stub_test_tracks.pkl')
    
    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])

    output_video_frame  = tracker.draw_annotations(video_frames, tracks)
    
    save_video(output_video_frame, "output_videos/goal2.avi")


if __name__ == "__main__":
    main()