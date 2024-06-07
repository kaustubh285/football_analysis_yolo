import cv2


def save_cropped_img(tracks, video_frames):
    for track_id, player in tracks["players"][0].items():
        bbox = player["bbox"]

        frame = video_frames[0]

        # crop bbox from frame
        cropped_img = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # save image
        cv2.imwrite(f"output/cropped_img.jpg", cropped_img)

        break
