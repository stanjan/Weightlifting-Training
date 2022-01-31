import cv2
import mediapipe as mp
import pose_recognition


class PoseDetector:

    def __init__(self, mode=False, upBody=False, smooth=True, complexity=1, detectionCon=0.5, trackingCon=0.5):
        self.file = None
        self.csv_writer = None
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.savedPoses = []
        self.lastPose = None
        self.id = 0

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, model_complexity=complexity,
                                     smooth_landmarks=self.smooth, min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackingCon)

    def process_pose(self, img, draw=True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        self.result = self.pose.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.landmarks = self.result.pose_landmarks
        if self.landmarks:
            self.lastPose = pose_recognition.landmarks_to_list(self.landmarks)
            self.savedPoses.append(self.lastPose)
        if draw and self.landmarks:
            self.mpDraw.draw_landmarks(img, self.landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
