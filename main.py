from utils import read_video, save_video
from tracking import Tracker

from team_assigner import TeamAssigner
import time


def main():
    file_name = "08fd33_4_short"
    video_frames, fps = read_video(f"input/{file_name}.mp4")
    tracker = Tracker("models/medium/best.pt")

    tracks = tracker.get_object_trac(
        video_frames, read_from_stub=True, stub_path=f"stubs/{file_name}.pkl"
    )

    # Assign Player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.assign_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )

            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, fps, f"output/{file_name}.avi")


if __name__ == "__main__":
    main()
