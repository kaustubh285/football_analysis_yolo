from utils import read_video, save_video
from tracking import Tracker


def main():
    video_frames = read_video("input/08fd33_4_full.mp4")
    tracker = Tracker("models/medium/best.pt")

    tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stub.pkl"
    )
    save_video(video_frames, "output/output.mp4")


if __name__ == "__main__":
    main()
