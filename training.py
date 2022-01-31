import datetime
import sys
import time
import cv2
import screeninfo as screeninfo
import PoseModule
import image_operations


COOLDOWN = 0.3
INITIAL_COOLDOWN = 3


class WeightliftingTraining:

    def __init__(self, data_path):
        self.data_path = data_path
        self.lifts = 0
        self.last_lifts = 0
        self.goal = 0
        self.is_lifted = False
        self.is_paused = False
        self.c_time = time.time()
        self.l_time = self.c_time
        self.stats = []

        self.load_data()
        self.display_task()

    def load_data(self):
        file = open(self.data_path, "r")
        lines = file.readlines()
        self.lifts = int(lines[0])
        self.last_lifts = self.lifts
        self.goal = int(lines[1])
        if len(lines) > 3:
            for data in lines[3].replace("\n", "").split(';'):
                data_array = data.split(',')
                lifts = int(data_array[0])
                day = datetime.datetime.fromtimestamp(float(data_array[1]))
                if day.date() != datetime.date.today():
                    self.stats.append([lifts, day])

        file.close()

    def save_data(self):
        file = open(self.data_path, "w")
        file.write(str(self.lifts))
        file.write("\n")
        file.write(str(self.goal))
        file.write("\n\n")
        for i, data in enumerate(self.stats):
            file.write(str(data[0]))
            file.write(",")
            file.write(str(datetime.datetime.timestamp(data[1])))
            file.write(";")
        file.write(str(self.lifts)+","+str(datetime.datetime.timestamp(datetime.datetime.today())))
        file.close()

    def lift(self, pose):
        if pose is None:
            self.l_time = self.c_time + INITIAL_COOLDOWN
            return None
        self.c_time = time.time()
        hook_y = pose[11][1]
        shifts = [pose[15][1] - hook_y, pose[16][1] - hook_y]

        if self.c_time - self.l_time > COOLDOWN:
            passing = 0
            if not self.is_lifted:
                for i, shift in enumerate(shifts):
                    if self.is_paused:
                        return None

                    if shift < 0.15:
                        passing += 1

                if passing == 2:
                    self.l_time = self.c_time
                    self.is_lifted = True
                    self.lifts += 1
                    self.save_data()
            else:
                for i, shift in enumerate(shifts):
                    if shift < -0.12:
                        self.l_time = self.c_time + INITIAL_COOLDOWN
                        self.is_paused = not self.is_paused
                    if self.is_paused:
                        sys.exit("Training finished")

                    if shift > 0.2:
                        passing += 1

                    if passing == 2:
                        self.l_time = self.c_time
                        self.is_lifted = False

    def display_task(self):
        window = "Weightlifting Training"
        screen = screeninfo.get_monitors()[0]
        cv2.namedWindow(window)
        vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

        camera_detector = PoseModule.PoseDetector(complexity=1)
        print("Video capture has started")
        while True:
            camera_success, camera_frame = vc.read()
            if not camera_success:
                break
            camera_frame = cv2.flip(camera_frame, 1)
            camera_frame = camera_detector.process_pose(camera_frame)
            self.lift(camera_detector.lastPose)
            camera_frame = image_operations.image_resize(camera_frame, height=screen.height - 50)

            final_frame = cv2.putText(camera_frame, str(self.lifts), (30, 130), 2, 4, color=(0, 0, 0), thickness=14)
            final_frame = cv2.putText(final_frame, str(self.lifts), (30, 130), 2, 4, color=(200 + self.is_lifted * 100, 200 + self.is_lifted * 100, 130 + (70*(not self.is_lifted)) + self.is_lifted * (100 * ((self.lifts%100)/15))), thickness=9)

            final_frame = self.bar(final_frame)

            cv2.moveWindow(window, int((screen.width-final_frame.shape[1])/2), 0)

            cv2.imshow(window, final_frame)
            key = cv2.waitKey(1)
            if key == 27:  # exit on ESC
                break

        vc.release()
        cv2.destroyWindow(window)

        self.save_data()

    def bar(self, frame):
        placement = (int(frame.shape[1]*0.05), int(frame.shape[0]-150))
        thickness = int(frame.shape[0]*0.022)
        progress = self.lifts / self.goal

        frame = cv2.rectangle(frame, placement, (int(placement[0] + frame.shape[1] * 0.88), int(placement[1]+frame.shape[0] * 0.07)), (255, 255, 255), lineType=cv2.LINE_AA, thickness=-1)
        frame = cv2.rectangle(frame, placement, (int(placement[0] + frame.shape[1] * 0.88), int(placement[1]+frame.shape[0] * 0.07)), (255, 255, 255),lineType=cv2.LINE_AA, thickness=thickness)

        tone = 255
        frame = cv2.rectangle(frame, placement, (int(placement[0] + frame.shape[1] * 0.88 * progress), int(placement[1] + frame.shape[0] * 0.07)), (0, tone, 0),lineType=cv2.LINE_AA, thickness=-1)
        frame = cv2.rectangle(frame, placement, (int(placement[0] + frame.shape[1] * 0.88 * progress), int(placement[1] + frame.shape[0] * 0.07)), (0, tone, 0),lineType=cv2.LINE_AA, thickness=thickness)

        tone -= 100
        for data in reversed(self.stats):
            last_progress = data[0] / self.goal
            frame = cv2.rectangle(frame, placement, (int(placement[0] + frame.shape[1] * 0.88 * last_progress), int(placement[1] + frame.shape[0] * 0.07)), (0, tone, 0), lineType=cv2.LINE_AA, thickness=-1)
            frame = cv2.rectangle(frame, placement, (int(placement[0] + frame.shape[1] * 0.88 * last_progress), int(placement[1] + frame.shape[0] * 0.07)), (0, tone, 0), lineType=cv2.LINE_AA, thickness=thickness)
            tone -= 20

        return frame


def main():
    training = WeightliftingTraining("training_data")


if __name__ == "__main__":
    main()
