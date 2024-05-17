import cv2


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames, fps


def save_video(frames, fps, path):
    print(fps)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(path, fourcc, 200, (frames[0].shape[1], frames[0].shape[0]))
    i = 0
    for frame in frames:
        print(i)
        i += 1
        out.write(frame)

    out.release()
