from utils import read_video, save_video
from tracking import Tracker


def main():
    video_frames, fps = read_video("input/08fd33_4_short.mp4")
    tracker = Tracker("models/medium/best.pt")

    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stub.pkl"
    )

    # output_video_frames = tracker.draw_annotations(video_frames, tracks)
    # save_video(output_video_frames, fps, "output/output.avi")


if __name__ == "__main__":
    main()
