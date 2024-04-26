from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner

def main():
    # read the video
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # initialize the tracker
    tracker = Tracker("models/tuned/best-tuned.pt")

    # get the tracks
    tracks = tracker.get_object_track(video_frames, read_from_stub=False, stub_path='stubs/stub_tracks.pkl')

    # interpolate the ball position
    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])

    # assign team colors 
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, player_track in player_tracks.items():
            team = team_assigner.assign_players_to_teams(video_frames[frame_num], player_track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # draw annotations
    output_video_frame  = tracker.draw_annotations(video_frames, tracks)

    # save output video
    save_video(output_video_frame, "output_videos/output-tuned.avi")


if __name__ == "__main__":
    main()