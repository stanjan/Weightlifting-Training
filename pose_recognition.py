import math
import numpy as np


def landmarks_to_list(landmarks):
    landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in landmarks.landmark]
    return landmarks


def normalize_pose(pose):
    if pose is None:
        return None
    normalized_pose = []
    centerXY = 0.5
    checkpoints = (11, 23)
    hx = 0
    hy = 11
    const_y = 0.2
    x1, y1, z = pose[checkpoints[0]]
    x2, y2, z = pose[checkpoints[1]]
    y = y1 - y2
    s = (y / const_y) * 100
    hook_x = centerXY
    hook_y = centerXY + (const_y / 2)
    for x, y, z in pose:
        dist_x = pose[hx][0] - x
        dist_y = pose[hy][1] - y
        sx = (100 * dist_x) / s
        sy = (100 * dist_y) / s
        normalized_pose.append([hook_x + sx, hook_y + sy, 0])

    return normalized_pose


def scale_number(value, min_value, max_value):
    if value <= min_value:
        return 0
    else:
        range = max_value - min_value
        scaled_value = value - min_value
        return scaled_value / range


def parse_angles(pose):
    angles = [calculate_angle([pose[11], pose[13], pose[15]]),
              calculate_angle([pose[13], pose[11], pose[23]]),
              calculate_angle([pose[12], pose[14], pose[16]]),
              calculate_angle([pose[14], pose[12], pose[24]])]
    return angles


def calculate_angle(three_landmarks):
    x1, y1, z = three_landmarks[0]
    x2, y2, z = three_landmarks[1]
    x3, y3, z = three_landmarks[2]
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle += 360
    return angle


MIN_ACCURACY = 0.8
MAX_ACCURACY = 1


def compare_poses(pose, source):  # Returns float value [0-1]. Higher the value, higher the accuracy
    if pose is None or source is None:
        return 0
    tracked_landmarks = (11, 12, 13, 14, 15, 16, 23, 24)
    landmarks_weights = (0, 0, 0.6, 0.6, 1, 1, 0, 0, 0.4, 0.4, 0.4, 0.4)

    landmarks = pose[:33]
    a_list = []
    for x1, y1, z1 in landmarks:
        index = pose.index([x1, y1, z1])
        if index not in tracked_landmarks:
            continue
        x2, y2, z2 = source[index]

        ax = 1 - abs(x1 - x2)
        ay = 1 - abs(y1 - y2)
        a_list.append(scale_number(np.average((ax, ay)), MIN_ACCURACY, 1))

    source_angles = source[35]
    pose_angles = parse_angles(pose)
    for angle1 in pose_angles:
        index = pose_angles.index(angle1)
        angle2 = source_angles[index]
        angle_accuracy = 1 - abs(angle2 - angle1)/angle2
        a_list.append(scale_number(angle_accuracy, MIN_ACCURACY, 1))

    return np.average(a_list, weights=landmarks_weights)
