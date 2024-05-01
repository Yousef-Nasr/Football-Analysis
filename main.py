from utils import read_video, save_video
from trackers import Tracker
import argparse
import numpy as np
from team_assigner import TeamAssigner
from player_assigner_ball import PlayerAssignerBall

import argparse

def main(args):
    # Read the video
    video_frames = read_video(args.input_video)

    # Initialize the tracker
    tracker = Tracker(args.model_weight)

    # Get the tracks
    tracks = tracker.get_object_track(video_frames, read_from_stub=args.read_from_stub, stub_path=args.stub_path)

    # Interpolate the ball position
    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])

    # Assign team colors
    team_assigner = TeamAssigner(method=args.clustering_method)
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, player_track in player_tracks.items():
            team = team_assigner.assign_players_to_teams(video_frames[frame_num], frame_num, player_track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    # Assign ball to player and get ball possession
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

    # Draw annotations
    output_video_frame  = tracker.draw_annotations(video_frames, tracks ,team_ball_control, team_assigner.team_colors)

    # Save output video
    save_video(output_video_frame, args.output_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Football Analysis')
    parser.add_argument('--input_video', type=str, default='input_videos/2e57b9_9.mp4', help='Path to the input video')
    parser.add_argument('--output_video', type=str, default='output_videos/output.avi', help='Path to save the output video')
    parser.add_argument('--clustering_method', type=str, default='center_box', choices=['center_box', 'foreground'], help='Clustering method for team assignment')
    parser.add_argument('--model_weight', type=str, default='models/tuned/best-tuned.pt', help='Path to the tracker model')
    parser.add_argument('--read_from_stub', action='store_true', help='Whether to read tracks from stub')
    parser.add_argument('--stub_path', type=str, default='stubs/stub_tracks.pkl', help='Path to the stub tracks')

    args = parser.parse_args()
    main(args)
