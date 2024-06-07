from utils import read_video, save_video, save_cropped_img
from tracking import Tracker


def main():
    video_frames, fps = read_video("input/08fd33_4_short.mp4")
    tracker = Tracker("models/medium/best.pt")

    tracks = tracker.get_object_trac(
        video_frames, read_from_stub=True, stub_path="stubs/track_stub.pkl"
    )

    save_cropped_img(tracks, video_frames)

    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, fps, "output/output2.avi")


if __name__ == "__main__":
    main()
