from utils import read_video, save_video
from trackers import Tracker


def main():
    video_frames = read_video("input_videos/goal2.mp4")
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_track(video_frames, read_from_stub=True, stub_path='stubs/stub_tracks.pkl')
    print(tracks)
    #save_video(video_frames, "output_videos/goalTest.avi")


if __name__ == "__main__":
    main()