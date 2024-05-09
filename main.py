from utils import read_video, save_video


def main():
    video_frames = read_video("input/08fd33_4_short.mp4")

    save_video(video_frames, "output/output.mp4")


if __name__ == "__main__":
    main()
