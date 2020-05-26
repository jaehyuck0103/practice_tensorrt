import json
import os

import cv2
import numpy as np

STATES = [
    "None",
    "Brake",
    "Left",
    "Brake Left",
    "Right",
    "Brake Right",
    "Emergency",
    "Brake Emergency",
]


def main():
    ROOT = "../Debug"

    num_frames = 5944
    offset = 7

    with open(os.path.join(ROOT, "result.json")) as fp:
        json_dict = json.load(fp)

    assert len(json_dict) == num_frames

    for idx in range(num_frames):
        print(idx)
        img1_path = os.path.join(ROOT, f"{idx}img.png")
        img2_path = os.path.join(ROOT, f"{idx}mask.png")

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # mask_image에 state 표시.
        if idx + offset < num_frames and json_dict[idx + offset]["result"]:
            for key, val in json_dict[idx + offset]["result"].items():
                state = STATES[val]
                xywh = None
                for i in range(offset):
                    if key in json_dict[idx + i]["bbox"]:
                        xywh = json_dict[idx + i]["bbox"][key]
                        break
                assert xywh is not None
                cv2.putText(
                    img2, state, (xywh[0], xywh[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2
                )

        img = np.concatenate([img1, img2], axis=0)

        if idx == 0:  # Video Setting
            H, W, _ = img.shape
            fourcc = cv2.VideoWriter_fourcc(*"X264")
            video_out = cv2.VideoWriter("movie.avi", fourcc, fps=10, frameSize=(W, H))

            if not video_out.isOpened():
                print("Output Video Open Failed")
                exit()

        video_out.write(img)

    video_out.release()
if __name__ == "__main__":
    main()
