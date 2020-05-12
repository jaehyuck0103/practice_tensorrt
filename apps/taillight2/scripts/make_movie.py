import os

import cv2
import numpy as np


def main():
    ROOT = "../Debug"

    for idx in range(5409):
        print(idx)
        img1_path = os.path.join(ROOT, f"{idx}img.png")
        img2_path = os.path.join(ROOT, f"{idx}mask.png")

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

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
