import json
import os

import cv2
import numpy as np

from preprocess_log import homo_tf

VOSS_PROJECT_PATH = "/mnt/SATA01/VoSS/20200316-174732(20191213-125018_emul)"
JSON_PATH = "./json.json"


def draw_cube(img, xy: np.array):
    """
    xy: (8, 2)
    """
    xy = xy.astype(int)
    pairs = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
    for i1, i2 in pairs:
        cv2.line(img, tuple(xy[i1]), tuple(xy[i2]), (0, 0, 255))


def project2img(xyz: np.array) -> np.array:
    """
    xyz: (N, 3)
    """
    RT = np.array(
        [
            [-0.005317, 0.003402, 0.999980, 1.624150],
            [-0.999920, -0.011526, -0.005277, 0.296660],
            [0.011508, -0.999928, 0.003463, 1.457150],
            [0, 0, 0, 1],
        ]
    )
    T_veh2cam = np.linalg.inv(RT)
    # ----
    RL = np.array(
        [
            [0.999844, 0.001632, -0.017567, 0],
            [-0.001632, 0.999999, 0, 0],
            [0.017567, 0.000029, 0.999846, 0],
            [0, 0, 0, 1],
        ]
    )
    T_cam2rect = np.linalg.inv(RL)
    # ----

    T_veh2rect = T_cam2rect @ T_veh2cam
    T_veh2rect = T_veh2rect[0:3, :]

    K = np.array(
        [
            [819.162645, 0.000000, 640.000000],
            [0.000000, 819.162645, 240.000000],
            [0.000000, 0.000000, 1.000000],
        ]
    )
    xy_projected = homo_tf(K @ T_veh2rect, xyz)
    return xy_projected


def rot_mat_Z3D(tz):
    return np.array(
        [
            [np.cos(tz), -np.sin(tz), 0],
            [np.sin(tz), np.cos(tz), 0],
            [0, 0, 1],
        ]
    )


def get_cube_xyz(xyz_center, lwh, rot):
    canonical_cube = get_canonical_cuboid(lwh)
    xyz = (rot_mat_Z3D(rot) @ canonical_cube.T).T
    xyz += xyz_center
    return xyz


def get_canonical_cuboid(sz):
    canonical_cuboid = 0.5 * np.array(
        [
            [-sz[0], +sz[1], -sz[2]],
            [+sz[0], +sz[1], -sz[2]],
            [-sz[0], +sz[1], +sz[2]],
            [+sz[0], +sz[1], +sz[2]],
            [-sz[0], -sz[1], -sz[2]],
            [+sz[0], -sz[1], -sz[2]],
            [-sz[0], -sz[1], +sz[2]],
            [+sz[0], -sz[1], +sz[2]],
        ]
    )
    return canonical_cuboid


def main():
    with open(JSON_PATH) as fp:
        json_list = json.load(fp)

    for each_time_dict in json_list:
        img_path = os.path.join(VOSS_PROJECT_PATH, each_time_dict["img_file"])
        img = cv2.imread(img_path)

        for each_obj in each_time_dict["objs"]:
            class_id, obj_id, x, y, z, l, w, h, rot = each_obj

            if x < 4 or abs(y) > 10:
                continue
            class_id = int(class_id)
            obj_id = int(obj_id)
            xyz_center = np.array([x, y, z])
            lwh = np.array([l, w, h])

            xyz_cube = get_cube_xyz(xyz_center, lwh, rot)
            xy_proj = project2img(xyz_cube)

            draw_cube(img, xy_proj)
            cv2.putText(
                img,
                str(class_id),
                tuple(xy_proj.astype(int)[0]),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
            )

        cv2.imshow("img", img)
        if cv2.waitKey() == ord("q"):
            break


if __name__ == "__main__":
    main()
