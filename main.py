from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_assigner_ball import PlayerAssignerBall

def main():
    # read the video
    video_frames = read_video("input_videos/mancity-passing.mp4")

    # initialize the tracker
    tracker = Tracker("models/tuned/best-tuned.pt")

    # get the tracks
    tracks = tracker.get_object_track(video_frames, read_from_stub=True, stub_path='stubs/stub_passing_tracks.pkl')

    # interpolate the ball position
    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])

    # assign team colors 
    team_assigner = TeamAssigner(method='foreground')
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, player_track in player_tracks.items():
            team = team_assigner.assign_players_to_teams(video_frames[frame_num], frame_num, player_track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    # assign ball to player and get ball possession
    ball_assigner = PlayerAssignerBall()
    team_ball_control = []
    for frame_num, player_tracks in enumerate(tracks['players']):
        try:
            ball_box = tracks['ball'][frame_num][1]['bbox']
            player_assigned = ball_assigner.assign_ball_to_player(player_tracks, ball_box)
        except:
            player_assigned = -1
        if player_assigned != -1:
            tracks['players'][frame_num][player_assigned]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][player_assigned]['team'])

        else:
            if len(team_ball_control) != 0 :
                team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)
    

    # draw annotations
    output_video_frame  = tracker.draw_annotations(video_frames, tracks ,team_ball_control, team_assigner.team_colors)

    # save output video
    save_video(output_video_frame, "output_videos/output-passing.avi")


if __name__ == "__main__":
    main()