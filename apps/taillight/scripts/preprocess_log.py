import json
import os

import numpy as np

import pandas as pd

VOSS_PROJECT_PATH = "/mnt/SATA01/VoSS/20200316-174732(20191213-125018_emul)"
CAMERA_LOG_PATH = os.path.join(VOSS_PROJECT_PATH, "cameras00.csv")
TRACKING_LOG_PATH = os.path.join(VOSS_PROJECT_PATH, "L-DET-RES.log")


def clip_angle(theta):
    return theta % (2 * np.pi)  # force in range [0, 2pi)


def homo_tf(A: np.array, x: np.array) -> np.array:
    """
    A: (?, D)
    x: (N, D-1)
    """
    assert A.ndim == 2 and x.ndim == 2
    assert A.shape[1] == x.shape[1] + 1
    assert 1 < A.shape[0] <= A.shape[1]

    x = x.T
    homo_x = np.vstack([x, np.ones(x.shape[1])])
    homo_y = A @ homo_x
    y = homo_y[:-1, ...] / homo_y[-1, ...]
    y = y.T

    return y


def tf2ego(ego_veh_pose: np.array, obj_poses: np.array) -> np.array:
    """
    ego_veh_pose: [ego_x, ego_y, ego_yaw]
    obj_poses: (N, 3)
    """
    assert ego_veh_pose.shape == (3,)
    assert obj_poses.ndim == 2 and obj_poses.shape[1] == 3

    delX, delY, delYaw = ego_veh_pose

    A = np.array(
        [
            [np.cos(delYaw), -np.sin(delYaw), delX],
            [np.sin(delYaw), np.cos(delYaw), delY],
            [0, 0, 1],
        ]
    )
    A = np.linalg.inv(A)

    # Transform (x, y)
    obj_poses[:, :2] = homo_tf(A, obj_poses[:, :2])
    # Transform yaw
    obj_poses[:, 2] = clip_angle(obj_poses[:, 2] - delYaw)

    return obj_poses


def main():
    cam_log = pd.read_csv(
        CAMERA_LOG_PATH, header=0, names=["time[ms]", "frame_idx"], index_col="time[ms]"
    )
    cam_log["img_file"] = cam_log["frame_idx"].astype(str).str.zfill(8)
    cam_log["img_file"] = "00/" + cam_log["img_file"] + ".png"
    assert cam_log.index.is_unique
    print(cam_log)

    tracking_log = pd.read_csv(
        TRACKING_LOG_PATH,
        header=0,
        names=["time[ms]", "class", "id", "x", "y", "heading", "w", "l", "h"],
        index_col="time[ms]",
    )
    tracking_log["z"] = tracking_log["h"] / 2  # temp

    # trackig_log에서 ego_poses만 추출
    ego_poses = tracking_log[tracking_log["class"] == -1]
    ego_poses = ego_poses[["x", "y", "heading"]]
    assert ego_poses.index.is_unique

    # ego_poses에 timestamp에 맞추어 png file 매칭
    matched_img_files = []
    for row in ego_poses.itertuples():
        min_idx = np.argmin(np.abs(cam_log.index.to_numpy() - row.Index))
        assert abs(cam_log.index[min_idx] - row.Index) < 30  # lidar, camera sync 확인 (30ms 이내)
        matched_img_files.append(cam_log["img_file"].iloc[min_idx])
    ego_poses["img_file"] = matched_img_files
    assert ego_poses["img_file"].is_unique
    print(ego_poses)

    # class_filtering - reserved
    print(tracking_log["class"].unique())
    obj_poses = tracking_log[tracking_log["class"].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
    assert obj_poses.index.unique().isin(ego_poses.index).all()

    # global -> ego coord
    json_list = []
    for row in ego_poses.itertuples():
        row_dict = {"img_file": row.img_file}

        ego_xyr = np.array([row.x, row.y, row.heading])

        # 1e-5 => one row나 empty row가 선택될 때도 dataframe이 return 될 수 있도록 하는 트릭
        obj_poses_each_time = obj_poses.loc[row.Index : row.Index + 1e-5, :].copy()
        obj_poses_each_time[["x", "y", "heading"]] = tf2ego(
            ego_xyr, obj_poses_each_time[["x", "y", "heading"]].to_numpy()
        )
        row_dict["objs"] = (
            obj_poses_each_time[["class", "id", "x", "y", "z", "l", "w", "h", "heading"]]
            .to_numpy()
            .tolist()
        )
        json_list.append(row_dict)

    with open("json.json", "w") as fp:
        json.dump(json_list, fp, indent=4)


if __name__ == "__main__":
    main()
