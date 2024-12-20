import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image as PILImage
import os
import mediapipe as mp
import sys
import time
import math
import socket
from Detector_Modules.HandDetectorModule import HandDetector
from Detector_Modules.PoseDetectorModule import PoseDetector

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
coefficient = 0.3
coefficient_body = 1.5

starting_pose = [0.04941637656952391,
                 -0.6935505764316097,
                 0.0835676500556497,
                 -2.2765016376491585,
                 -0.03950194350709957,
                 1.6093278345679556,
                 0.7534486589746342]

class HandGesture3DReconstruction:
    def __init__(self, folder_path):
        # Initialize attributes for 3D reconstruction
        self.all_colors_rgb = None
        self.folder_path = folder_path
        self.all_points3D = []
        self.all_colors = []
        self.K = np.array([[1249, 0, 0],
                           [0, 1283, 0],
                           [0, 0, 1]])  # Intrinsic camera matrix
        self.vertices = None
        self.faces = None
        self.azimuth = 0
        self.elevation = 0
        self.scale_factor = 1.0
        self.x_min = 20
        self.x_mid = 35
        self.x_max = 180
        self.palm_angle_min = -50
        self.palm_angle_max = 50
        self.palm_angle_mid = 20
        self.y_min = 20
        self.y_mid = 35
        self.y_max = 180
        self.z_min = 20
        self.z_mid = 35
        self.z_max = 180
        self.claw_open_angle = 60
        self.claw_close_angle = 0
        self.wrist_y_min = 0.3
        self.wrist_y_max = 0.9
        self.fist_threshold = 7
        self.base_angle_min = 20
        self.base_angle_max = 50
        self.elbow_angle_min = 20
        self.elbow_angle_max = 40
        self.arm_angle_min = 20
        self.arm_angle_max = 50
        self.finger_angle_min = 20
        self.finger_angle_max = 50
        self.azimuth_min = 20
        self.azimuth_max = 60
        self.plam_size_min = 10
        self.plam_size_max = 100
        self.servo_angles = [0] *6
        self.servo_angles[1] = 60
        self.servo_angles[2] = 60

        self.cap = cv2.VideoCapture(0)
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ani = None


        host = '10.71.36.55'  # Localhost or use your server"s IP address
        port = 8000  # Port to listen on (non-privileged ports are > 1023)

        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.bind((host, port))
            self.server.listen(1)
            print(f"TCP server is listening on {host}:{port}...")
            self.client, addr = self.server.accept()
            print(f"Connection accepted from {addr}")
            formatted_str = str(self.servo_angles)
            self.client.send(bytes(formatted_str, 'UTF-8'))
        except Exception as e:
            print(f"An error occurred in TCP Connection Module: {e}")
            sys.exit()
    def compute_projection_matrices(self, K, R, t):
        proj_matrix1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        proj_matrix2 = np.hstack((R, t.reshape(-1, 1)))
        proj_matrix1 = K @ proj_matrix1
        proj_matrix2 = K @ proj_matrix2
        return proj_matrix1, proj_matrix2

    def process_image_pair(self, img1, img2, K):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        if descriptors1 is None or descriptors2 is None:
            print("Warning: No descriptors found in one of the images")
            return None, None

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        inlier_count = np.sum(mask)
        print(f"Number of inliers after essential matrix calculation: {inlier_count}")

        points1 = points1[mask.ravel() == 1]
        points2 = points2[mask.ravel() == 1]

        if points1.shape[0] == 0 or points2.shape[0] == 0:
            print("Error: Points arrays are empty after inlier filtering")
            return None, None

        _, R, t, pose_mask = cv2.recoverPose(E, points1, points2, K)
        pose_inlier_count = np.sum(pose_mask)
        print(f"Number of inliers after pose recovery: {pose_inlier_count}")

        pose_mask_binary = (pose_mask == 255).astype(np.uint8)
        print(f"Pose mask binary unique values: {np.unique(pose_mask_binary)}")

        filtered_points1 = points1[pose_mask_binary.ravel() == 1]
        filtered_points2 = points2[pose_mask_binary.ravel() == 1]

        if filtered_points1.shape[0] == 0 or filtered_points2.shape[0] == 0:
            print("Error: Points arrays are empty after pose recovery filtering")
            return None, None

        filtered_points1 = filtered_points1.T
        filtered_points2 = filtered_points2.T

        if filtered_points1.shape[0] == 2 and filtered_points2.shape[0] == 2:
            proj_matrix1, proj_matrix2 = self.compute_projection_matrices(K, R, t)

            try:
                points4D = cv2.triangulatePoints(proj_matrix1, proj_matrix2, filtered_points1, filtered_points2)
                points4D /= points4D[3]
                points3D = points4D[:3].T

                # Use colors from the first image only
                colors = np.array([img1[int(p[1]), int(p[0])] for p in filtered_points1.T])
                return points3D, colors
            except cv2.error as e:
                print(f"Error during triangulation: {e}")
                return None, None
        else:
            print("Error: Points arrays should have shape (2, N)")
            return None, None

    def distance_to_azimuth_angle(self, thumb_tip, index_tip):
        # Compute azimuth angle from the distance or any other method you are using
        azimuth = np.arctan2(thumb_tip.y - index_tip.y, thumb_tip.x - index_tip.x) * (180 / np.pi)
        azimuth = clamp(azimuth, self.azimuth_min, self.azimuth_max)  # Clamp to desired range
        return azimuth

    def is_fist(self, hand_landmarks, palm_size):
        distance_sum = 0
        WRIST = hand_landmarks.landmark[0]
        for i in [7, 8, 11, 12, 15, 16, 19, 20]:
            distance_sum += ((WRIST.x - hand_landmarks.landmark[i].x) ** 2 +
                             (WRIST.y - hand_landmarks.landmark[i].y) ** 2 +
                             (WRIST.z - hand_landmarks.landmark[i].z) ** 2) * 0.5
        return distance_sum / palm_size < self.fist_threshold

    def read_images_from_folder(self):
        image_files = [os.path.join(self.folder_path, f) for f in sorted(os.listdir(self.folder_path)) if
                       f.endswith(('jpg', 'png', 'jpeg'))]
        images = [cv2.imread(img_file) for img_file in image_files]  # Read color images
        return images

    def map_value(self, value, left_min, left_max, right_min, right_max):
        # Maps value from one range to another
        left_span = left_max - left_min
        right_span = right_max - right_min
        value_scaled = float(value - left_min) / float(left_span)
        return right_min + (value_scaled * right_span)

    def normalize(self, value, min_value, max_value, new_min, new_max):
        """
        Normalize the given value to a new range.

        :param value: The value to be normalized.
        :param min_value: The minimum value of the original range.
        :param max_value: The maximum value of the original range.
        :param new_min: The minimum value of the new range.
        :param new_max: The maximum value of the new range.
        :return: The normalized value.
        """
        # Ensure the value is within the original range
        value = max(min_value, min(value, max_value))

        # Normalize the value to the range 0 to 1
        normalized_value = (value - min_value) / (max_value - min_value)

        # Scale the normalized value to the new range
        return new_min + (normalized_value * (new_max - new_min))

    def find_robot_angles(self, fps_cap=60, show_fps=True, source=0):
        """
        Capture webcam video from the specified "source" (default is 0) using the opencv VideoCapture function.
        It's possible to cap/limit the number of FPS using the "fps_cap" variable (default is 60) and to show the actual FPS footage (shown by default).
        The program stops if "q" is pressed or there is an error in opening/using the capture source.

        :param: fps_cap (int)
            max framerate allowed (default is 60)
        :param: show_fps (bool)
            shows a real-time framerate indicator (default is True)
        :param: source (int)
            select the webcam source number used in OpenCV (default is 0)
        """
        assert fps_cap >= 1, f"fps_cap should be at least 1\n"
        assert source >= 0, f"source needs to be greater or equal than 0\n"

        # instantiation of the HandDetector and PoseDetector
        HandDet = HandDetector()
        PoseDet = PoseDetector(detCon=0.7, trackCon=0.7, modCompl=1)

        cv2.setUseOptimized(True)

        ctime = 0  # current time (used to compute FPS)
        ptime = 0  # past time (used to compute FPS)
        prev_time = 0  # previous time variable, used to set the FPS limit

        fps_lim = fps_cap  # FPS upper limit value, needed for estimating the time for each frame and increasing performances

        time_lim = 1. / fps_lim  # time window for each frame taken by the webcam

        # capture the video from the webcam
        if not self.cap.isOpened():  # if the camera can't be opened exit the program
            print("Cannot open camera")
            exit()
        i = 0
        while i<2:
            i=i+1
            # computed delta time for FPS capping
            delta_time = time.perf_counter() - prev_time
            ret, self.frame = self.cap.read()  # read a frame from the webcam
            if not ret:  # if a frame can't be read, exit the program
                print("Can't receive frame from camera/stream end")
                break
            # if frame comes from a webcam flip it to mirror the image
            if source == 0:
                self.frame = cv2.flip(self.frame, 1)
            if delta_time >= time_lim:  # if the time passed is bigger or equal than the frame time, process the frame
                prev_time = time.perf_counter()
                # compute the actual frame rate per second (FPS) of the webcam video capture stream, and show it
                ctime = time.perf_counter()
                fps = 1.0 / float(ctime - ptime)
                ptime = ctime
                # find the hands in the image
                self.frame = HandDet.findHands(frame=self.frame, draw=True)
                # find the pose in the image
                frame_pose = PoseDet.findPose(frame=self.frame, draw=False)
                # extract the pose 2D and 3D keypoints from the frame
                lm_list_pose = PoseDet.findPosePosition(
                    self.frame, additional_info=True, draw=False)
                lm_3dlist_pose = PoseDet.find3DPosePosition()
                # extract the hand 2D keypoints from the frame
                hand_lmlist, self.frame = HandDet.findHandPosition(
                    frame=self.frame, hand_num=0, draw=False)
                # If all the keypoints data is available, compute the pose angles, the hand aperture and send it to the robot
                if len(lm_list_pose) > 0 and len(lm_3dlist_pose) > 0 and len(hand_lmlist) > 0:
                    elbow_angle_3d = PoseDet.findAngle(
                        self.frame, 12, 14, 16, angle3d=True, draw=True)
                    elbow_angle_3d_second = PoseDet.findAngle(
                        self.frame, 11, 13, 15, angle3d=True, draw=True)
                    elbow_angle_3d_body = PoseDet.findAngle(
                        self.frame, 23, 11, 12, angle3d=True, draw=True)
                    self.frame, aperture = HandDet.findHandAperture(
                        frame=self.frame, verbose=True, show_aperture=True)
                    # creating a valid angle list for the robot joint, given the extracted angles and hand aperture
                    elbow_angle_3d_body = self.normalize(elbow_angle_3d_body,70, 90, 30, 150)
                    elbow_angle_3d = self.normalize(elbow_angle_3d_body,0, 180, 40, 130)
                    elbow_angle_3d_second = self.normalize(elbow_angle_3d_second,0, 360, 0, 360)
                    self.azimuth = elbow_angle_3d_second
                    steps_per_rev = 4096
                    deg_per_step = 360/steps_per_rev
                    steps = elbow_angle_3d_second/deg_per_step

                    self.servo_angles = [0, elbow_angle_3d_body, elbow_angle_3d, 0, elbow_angle_3d_second, steps]
                    print(self.servo_angles)
                    formatted_str = str(self.servo_angles)
                    self.client.send(bytes(formatted_str, 'UTF-8'))
                    # vals = [starting_pose[0] + coefficient_body * ((elbow_angle_3d_body - 80) / 10),
                    #         (starting_pose[1] + coefficient * ((elbow_angle_3d_second - 90) / 10)),
                    #         starting_pose[2],
                    #         (starting_pose[3] + coefficient *
                    #          ((elbow_angle_3d - 90) / 10)),
                    #         starting_pose[4],
                    #         (starting_pose[5] + coefficient * (aperture / 10)),
                    #         starting_pose[6]]
                    # self.servo_angles = vals
                if show_fps:
                    cv2.putText(self.frame, "FPS:" + str(round(fps, 0)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2,
                                (255, 255, 255), 1)

                azimuth_rad = np.radians(self.azimuth)
                elevation_rad = np.radians(self.elevation)

                R_azimuth = np.array([[np.cos(azimuth_rad), -np.sin(azimuth_rad), 0],
                                      [np.sin(azimuth_rad), np.cos(azimuth_rad), 0],
                                      [0, 0, 1]])
                R_elevation = np.array([[1, 0, 0],
                                        [0, np.cos(elevation_rad), -np.sin(elevation_rad)],
                                        [0, np.sin(elevation_rad), np.cos(elevation_rad)]])
                rotated_vertices = self.vertices.dot(R_azimuth).dot(R_elevation) * self.scale_factor

                # Save the scatter plot without displaying it
                self.ax.clear()
                self.all_colors = np.array(self.all_colors, dtype=np.float64)
                self.all_colors_rgb = self.all_colors[:, [2, 1, 0]] / 255.0
                self.ax.scatter(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2],
                                c=self.all_colors_rgb, marker='.')
                self.ax.xaxis.set_visible(False)
                self.ax.yaxis.set_visible(False)
                self.ax.zaxis.set_visible(False)
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.ax.set_zticks([])
                self.ax.set_xlabel('')
                self.ax.set_ylabel('')
                self.ax.set_zlabel('')
                self.ax.set_title('')
                self.ax.grid(False)
                self.ax.set_facecolor('none')
                scatter_plot_filename = 'scatter_plot.png'
                self.fig.savefig(scatter_plot_filename, bbox_inches='tight', pad_inches=0, dpi=300)

                # Read the saved scatter plot image
                scatter_plot_image = cv2.imread(scatter_plot_filename)
                image_rgb = cv2.cvtColor(cv2.resize(self.frame, (650, 650)), cv2.COLOR_RGB2BGR)
                height = image_rgb.shape[0]
                width = int((height / scatter_plot_image.shape[0]) * scatter_plot_image.shape[1])
                scatter_plot_resized = cv2.resize(scatter_plot_image, (width // 2, height))
                video_frame_resized = cv2.resize(image_rgb, (width // 2, height))
                video_frame_resized_rgb = cv2.cvtColor(video_frame_resized, cv2.COLOR_BGR2RGB)
                combined_image = np.hstack((scatter_plot_resized, video_frame_resized_rgb))
                cv2.namedWindow('Combined View', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('Combined View', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Combined View', combined_image)
                # Exit the loop after processing the first frame to avoid frequent camera on/off
                # break

        # self.cap.release()
        # cv2.destroyAllWindows()
        return

    def rotate(self, frame):
        self.find_robot_angles(10, True, 0)
        # Continue with 3D plot and display logic


    def init(self):
        images = self.read_images_from_folder()

        self.all_points3D = []
        self.all_colors = []
        self.K[0, 2] = images[0].shape[1] / 2
        self.K[1, 2] = images[0].shape[0] / 2
        for i in range(len(images) - 1):
            points3D, colors = self.process_image_pair(images[i], images[i + 1], self.K)
            if points3D is not None:
                self.all_points3D.append(points3D)
                self.all_colors.append(colors)
        if self.all_points3D:
            self.all_points3D = np.vstack(self.all_points3D)
            self.all_colors = np.vstack(self.all_colors)  # Aggregate colors
            print(f"Total number of 3D points: {self.all_points3D.shape[0]}")
            self.vertices, self.faces = self.all_points3D, []  # Placeholder for faces
            self.ani = FuncAnimation(self.fig, self.rotate, interval=100, cache_frame_data=False)
            plt.show()


folder_path = "E:\\Indumathi\\Multiview-3D-Reconstruction-main\\Multiview-3D-Reconstruction-main\\Received_images"
reconstruction = HandGesture3DReconstruction(folder_path)
reconstruction.init()
Final_Altogether_Folder_Read.py
Displaying Final_Altogether_Folder_Read.py.
