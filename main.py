from utils import read_video, save_video
from tracking import Tracker
from player_ball_assigner import PlayerBallAssigner
from team_assigner import TeamAssigner
import numpy as np
import time


def main():
    file_name = "football_match_1"
    tracker = Tracker("models/medium/best.pt")
    read_from_stub = True
    stub_path = f"stubs/{file_name}.pkl"
    output_path = f"output/{file_name}.avi"

    video_frames, fps = read_video(f"input/{file_name}.mp4")
    tracks = tracker.get_object_trac(
        video_frames, read_from_stub=read_from_stub, stub_path=stub_path
    )

    # fill missing ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign Player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):

        # For team assignment
        for player_id, track in player_track.items():
            team = team_assigner.assign_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )

            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

        # For ball assignment
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][assigned_player]["team"]
            )
        else:
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )

    save_video(output_video_frames, fps, output_path)


if __name__ == "__main__":
    main()
    # 3:03
