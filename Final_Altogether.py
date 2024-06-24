from matplotlib.animation import FuncAnimation
import cv2
import socket
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from tqdm import tqdm
import os
import time
from tomlkit import boolean
import mediapipe as mp

import requests
from PIL import Image
import os
from io import BytesIO
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime
import logging

from cvzone.HandTrackingModule import HandDetector

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']

# Specify the directory to save the images
SAVE_DIR = r'E:\Indumathi\Multiview-3D-Reconstruction-main\Multiview-3D-Reconstruction-main\Retrieved_from_Google_photos'

logging.basicConfig(level=logging.INFO)



# server = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
# mac = "A4:F9:33:A9:81:F1"

# mac = "CC:15:31:2F:ED:28"
# Other systme's IP address, say CC:15:31:2F:ED:28 will become File "D:\Bluetooth_Alone\Serving.py", line 8, in <module>
#     server.bind((mac,prt))
# OSError: [WinError 10049] The requested address is not valid in its context

# Address in this format A4-F9-33-A9-81-F1, it will say   File "D:\Bluetooth_Alone\Serving.py", line 11, in <module>
#     server.bind((mac,prt))
# OSError: bad bluetooth address

# "A4:F9:33:A9:81:F1" is the only accepted format and also it is of same machine as it is a server


clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
map_range = lambda x, in_min, in_max, out_min, out_max: abs((x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min)
cam_source = 0

cap = cv2.VideoCapture(cam_source)
write_video = True

# # video writer
# if write_video:
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))
# class ImageLoader:
#     def __init__(self, img_dir: str, downscale_factor: float):
#         with open(os.path.join(img_dir, 'K.txt')) as f:
#             self.K = np.array([[float(x) for x in line.split()] for line in f])
#             self.image_list = [os.path.join(img_dir, image) for image in sorted(os.listdir(img_dir)) if
#                                image.lower().endswith(('.jpg', '.png', 'jpeg'))]
#         self.path = os.getcwd()
#         self.factor = downscale_factor
#         self.downscale()
#
#
#     def downscale(self):
#         self.K[:2, :] /= self.factor
#
#     def downscale_image(self, image):
#         for _ in range(int(self.factor / 2)):
#             image = cv2.pyrDown(image)
#         return image

class Sfm:
    def __init__(self, img_dir, scale_factor=1, azimuth=0, elevation=0, downscale_factor=2.0, ):
#        self.img_obj = ImageLoader(img_dir, downscale_factor)
#        self.downscale_factor = downscale_factor
        # self.fig = plt.figure(figsize=(10, 5))  # Initialize the figure
        # self.ax = self.fig.add_subplot(121, projection='3d',position=[0.05, 0.1, 0.4, 0.8])  # Adjust position as needed
        # self.ax1 = self.fig.add_subplot(122, position=[0.55, 0.1, 0.4, 0.4])  # Adjust position as needed
        self.fig = plt.figure(figsize=(12, 6))
        self.ax = self.fig.add_subplot(121, projection='3d', position=[0.05, 0.1, 0.4, 0.8])
        self.azimuth = azimuth
        self.elevation = elevation
        self.scale_factor = scale_factor
        self.detector = HandDetector(detectionCon=0.7, maxHands=2)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.vertices = None
        self.faces = None
        self.cmd = 'S'
        self.x_min = 0
        self.x_mid = 75
        self.x_max = 150
        # use angle between wrist and index finger to control x axis
        self.palm_angle_min = -50
        self.palm_angle_mid = 20

        self.y_min = 0
        self.y_mid = 90
        self.y_max = 180
        # use wrist y to control y axis
        self.wrist_y_min = 0.3
        self.wrist_y_max = 0.9

        self.z_min = 10
        self.z_mid = 90
        self.z_max = 180
        # use palm size to control z axis
        self.plam_size_min = 0.1
        self.plam_size_max = 0.3

        self.claw_open_angle = 60
        self.claw_close_angle = 0

#        self.servo_angle = [self.x_mid, self.y_mid, self.z_mid, self.claw_open_angle]  # [x, y, z, claw]
#        self.prev_servo_angle = self.servo_angle
        self.fist_threshold = 7

        self.servo_angles = [0] * 6  # List to store servo angles for 6 servos
        self.prev_servo_angle = [0] * 6  # To store previous servo angles
        self.servo_angle = [0] * 6  # Current servo angles
        # port = 4
        # serverMACAddress = 'A4-F9-33-A9-81-F1'  # Bluetooth MAC address
        # try:
        #     self.server = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        #     self.server.connect((serverMACAddress, port))
        # except Exception as e:
        #     print(f"An error occurred in Bluetooth Connection Module: {e}")
        #    sys.exit()  # Terminate the program if an error occurs during the connection attempt

    def authenticate(self):
        """Authenticates with Google and returns credentials."""
        creds = None
        token_path = 'token.json'
        credentials_path = 'path/to/client_secret.json'
        SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']

        try:
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        credentials_path,
                        scopes=SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                # Save the credentials for the next run
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())

            return creds

        except Exception as e:
            logging.error(f"Authentication error: {e}")
            return None

    def list_photos(self, credentials):
        """Lists all photos in Google Photos."""
        service = build(
            'photoslibrary',
            'v1',
            credentials=credentials,
            discoveryServiceUrl='https://photoslibrary.googleapis.com/$discovery/rest',
            cache_discovery=False
        )

        photos = []
        # Call the Google Photos API to list all media items
        response = service.mediaItems().list().execute()
        items = response.get('mediaItems', [])
        # Extract image URLs and creation times
        for item in items:
            if 'image' in item['mimeType']:
                url = item['baseUrl'] + '=w2048-h1024'
                creation_time = datetime.strptime(item['mediaMetadata']['creationTime'], "%Y-%m-%dT%H:%M:%S.%fZ")
                photos.append({'url': url, 'creation_time': creation_time})
        # Sort photos based on creation time
        photos.sort(key=lambda x: x['creation_time'], reverse=True)
        return photos

    def load_image_from_url(self, url, filename):
        """Loads an image from the given URL and saves it to the specified directory."""
        response = requests.get(url)
        img_data = response.content
        img = Image.open(BytesIO(img_data))

        # Save the image to the specified directory
        img.save(os.path.join(SAVE_DIR, filename))

        return img

    def process_images(self, credentials):
        """Loads and processes images from Google Photos."""
        # List all photos from Google Photos
        photos = self.list_photos(credentials)

        # Save each photo with the appropriate filename
        for idx, photo in enumerate(photos, start=1):
            filename = f'img{idx}.jpg'  # Define filename based on index
            self.load_image_from_url(photo['url'], filename)

    def triangulation(self, point_2d_1, point_2d_2, projection_matrix_1, projection_matrix_2) -> tuple:
        '''
        Triangulates 3d points from 2d vectors and projection matrices
        returns projection matrix of first camera, projection matrix of second camera, point cloud
        '''
        pt_cloud = cv2.triangulatePoints(point_2d_1, point_2d_2, projection_matrix_1.T, projection_matrix_2.T)
        return projection_matrix_1.T, projection_matrix_2.T, (pt_cloud / pt_cloud[3])

    def PnP(self, obj_point, image_point , K, dist_coeff, rot_vector, initial) ->  tuple:
        '''
        Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
        returns rotational matrix, translational matrix, image points, object points, rotational vector
        '''
        if initial == 1:
            obj_point = obj_point[:, 0 ,:]
            image_point = image_point.T
            rot_vector = rot_vector.T
        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
        # Converts a rotation matrix to a rotation vector or vice versa
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

        if inlier is not None:
            image_point = image_point[inlier[:, 0]]
            obj_point = obj_point[inlier[:, 0]]
            rot_vector = rot_vector[inlier[:, 0]]
        return rot_matrix, tran_vector, image_point, obj_point, rot_vector

    def reprojection_error(self, obj_points, image_points, transform_matrix, K, homogenity) ->tuple:
        '''
        Calculates the reprojection error ie the distance between the projected points and the actual points.
        returns total error, object points
        '''
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        if homogenity == 1:
            obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)
        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])
        total_error = cv2.norm(image_points_calc, np.float32(image_points.T) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
        return total_error / len(image_points_calc), obj_points

    def optimal_reprojection_error(self, obj_points) -> np.array:
        '''
        calculates of the reprojection error during bundle adjustment
        returns error
        '''
        transform_matrix = obj_points[0:12].reshape((3,4))
        K = obj_points[12:21].reshape((3,3))
        rest = int(len(obj_points[21:]) * 0.4)
        p = obj_points[21:21 + rest].reshape((2, int(rest/2))).T
        obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:])/3), 3))
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points = image_points[:, 0, :]
        error = [ (p[idx] - image_points[idx])**2 for idx in range(len(p))]
        return np.array(error).ravel()/len(p)

    def bundle_adjustment(self, _3d_point, opt, transform_matrix_new, K, r_error) -> tuple:
        '''
        Bundle adjustment for the image and object points
        returns object points, image points, transformation matrix
        '''
        opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
        opt_variables = np.hstack((opt_variables, opt.ravel()))
        opt_variables = np.hstack((opt_variables, _3d_point.ravel()))

        values_corrected = least_squares(self.optimal_reprojection_error, opt_variables, gtol = r_error).x
        K = values_corrected[12:21].reshape((3,3))
        rest = int(len(values_corrected[21:]) * 0.4)
        return values_corrected[21 + rest:].reshape((int(len(values_corrected[21 + rest:])/3), 3)), values_corrected[21:21 + rest].reshape((2, int(rest/2))).T, values_corrected[0:12].reshape((3,4))

    def to_ply(self, path, point_cloud, colors):
        try:
            # Reshape point cloud and colors
            out_points = point_cloud.reshape(-1, 3) * 200
            out_colors = colors.reshape(-1, 3)

            # Concatenate points and colors into vertices
            verts = np.hstack([out_points, out_colors])

            # Calculate mean of vertices
            mean = np.mean(verts[:, :3], axis=0)

            # Center vertices around mean
            scaled_verts = verts[:, :3] - mean

            # Calculate distance of vertices from the mean
            dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)

            # Filter vertices based on distance threshold
            indx = np.where(dist < np.mean(dist) + 300)
            verts = verts[indx]

            # Assuming you have a method to generate faces from vertices
            faces = None
            return verts, faces

        except Exception as e:
            print("An error occurred:", e)
            return None, None

    def common_points(self, image_points_1, image_points_2, image_points_3) -> tuple:
        '''
        Finds the common points between image 1 and 2 , image 2 and 3
        returns common points of image 1-2, common points of image 2-3, mask of common points 1-2 , mask for common points 2-3
        '''
        cm_points_1 = []
        cm_points_2 = []
        for i in range(image_points_1.shape[0]):
            a = np.where(image_points_2 == image_points_1[i, :])
            if a[0].size != 0:
                cm_points_1.append(i)
                cm_points_2.append(a[0][0])

        mask_array_1 = np.ma.array(image_points_2, mask=False)
        mask_array_1.mask[cm_points_2] = True
        mask_array_1 = mask_array_1.compressed()
        mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

        mask_array_2 = np.ma.array(image_points_3, mask=False)
        mask_array_2.mask[cm_points_2] = True
        mask_array_2 = mask_array_2.compressed()
        mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)
 #       print(" Shape New Array", mask_array_1.shape, mask_array_2.shape)
        return np.array(cm_points_1), np.array(cm_points_2), mask_array_1, mask_array_2

    def find_features(self, image_0, image_1) -> tuple:
        '''
        Feature detection using the sift algorithm and KNN
        return keypoints(features) of image1 and image2
        '''

        sift = cv2.SIFT_create()
        key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        feature = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                feature.append(m)

        return np.float32([key_points_0[m.queryIdx].pt for m in feature]), np.float32([key_points_1[m.trainIdx].pt for m in feature])

    def is_fist(self,hand_landmarks, palm_size):
        # calculate the distance between the wrist and the each finger tip
        distance_sum = 0
        WRIST = hand_landmarks.landmark[0]
        for i in [7, 8, 11, 12, 15, 16, 19, 20]:
            distance_sum += ((WRIST.x - hand_landmarks.landmark[i].x) ** 2 + \
                             (WRIST.y - hand_landmarks.landmark[i].y) ** 2 + \
                             (WRIST.z - hand_landmarks.landmark[i].z) ** 2) ** 0.5
        return distance_sum / palm_size < self.fist_threshold

    def map_value(self, value, left_min, left_max, right_min, right_max):
        # Maps value from one range to another
        left_span = left_max - left_min
        right_span = right_max - right_min
        value_scaled = float(value - left_min) / float(left_span)
        return right_min + (value_scaled * right_span)

    def landmark_to_servo_angles(self, hand_landmarks):
        # Map hand landmarks to servo angles for the robot arm
        hip_angle = self.map_value(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x, 0, 1, 0, 180)
        shoulder_angle = self.map_value(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_CMC].x, 0, 1, 0, 180)
        elbow_angle = self.map_value(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].x, 0, 1, 0, 180)
        wrist_angle = self.map_value(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].x, 0, 1, 0, 180)
        finger_angle = self.map_value(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP].x, 0, 1, 0, 180)
        return [hip_angle, shoulder_angle, elbow_angle, wrist_angle, finger_angle]

    def distance_to_azimuth_angle(self, thumb_tip, index_tip):
        # Compute the distance between the tip of the thumb and the tip of the index finger
        distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2 + (thumb_tip.z - index_tip.z) ** 2)
        # Map this distance to an azimuth angle (adjust range as needed)
        azimuth_angle = self.map_value(distance, 0, 0.1, -90, 90)  # Mapping distance to azimuth angle (-90° to 90°)
        return azimuth_angle, 0

    def rotate(self, frame):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x > 0.5:
                            self.servo_angles[:5] = self.landmark_to_servo_angles(hand_landmarks)
                        else:
                            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            self.azimuth, self.elevation = self.distance_to_azimuth_angle(thumb_tip, index_tip)

                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                # Perform azimuth and elevation angle calculations
                azimuth_rad = np.radians(self.azimuth)
                elevation_rad = np.radians(self.elevation)

                # Define rotation matrices
                R_azimuth = np.array([[np.cos(azimuth_rad), -np.sin(azimuth_rad), 0],
                                      [np.sin(azimuth_rad), np.cos(azimuth_rad), 0],
                                      [0, 0, 1]])
                R_elevation = np.array([[1, 0, 0],
                                        [0, np.cos(elevation_rad), -np.sin(elevation_rad)],
                                        [0, np.sin(elevation_rad), np.cos(elevation_rad)]])

                # Assuming you have defined self.vertices and self.faces elsewhere
                # Apply rotations and scaling to vertices
                rotated_vertices = self.vertices.dot(R_azimuth).dot(R_elevation) * self.scale_factor

                # Plotting code (assuming you are using matplotlib)
                # Replace with your specific plotting setup
                # Clear the previous plot
                self.ax.clear()

                # Plot vertices
                self.ax.scatter(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2], c='b', marker='.')

                # Plot the mesh
                self.ax.plot_trisurf(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2],
                                     triangles=self.faces, color='blue')

                self.ax.axis('off')

                # Redraw the plot
                self.fig.canvas.draw()

                cv2.imshow('MediaPipe Hands', image)

                if write_video:
                    out.write(image)

                if cv2.waitKey(5) & 0xFF == 27:
                    if write_video:
                        out.release()
                    break

        cap.release()
        cv2.destroyAllWindows()
    #
    # def landmark_to_servo_angle(self,hand_landmarks, which):
    #     self.servo_angle = [self.x_mid, self.y_mid, self.z_mid, self.claw_open_angle]
    #     WRIST = hand_landmarks.landmark[0]
    #     INDEX_FINGER_MCP = hand_landmarks.landmark[5]
    #     # calculate the distance between the wrist and the index finger
    #     palm_size = ((WRIST.x - INDEX_FINGER_MCP.x) ** 2 + (WRIST.y - INDEX_FINGER_MCP.y) ** 2 + (
    #                 WRIST.z - INDEX_FINGER_MCP.z) ** 2) ** 0.5
    #
    #     if self.is_fist(hand_landmarks, palm_size):
    #         self.servo_angle[3] = self.claw_close_angle
    #     else:
    #         self.servo_angle[3] = self.claw_open_angle
    #
    #     # calculate x angle
    #     distance = palm_size
    #     angle = (WRIST.x - INDEX_FINGER_MCP.x) / distance  # calculate the radian between the wrist and the index finger
    #     angle = int(angle * 180 / 3.1415926)  # convert radian to degree
    #     angle = clamp(angle, self.palm_angle_min, self.palm_angle_mid)
    #     if(which == 'left'):
    #         rot_angle = map_range(angle, self.palm_angle_min, self.palm_angle_mid, self.x_max, self.x_min)
    #
    #     if(which == 'right'):
    #         self.servo_angle[0] = map_range(angle, self.palm_angle_min, self.palm_angle_mid, self.x_max, self.x_min)
    #         # calculate y angle
    #         wrist_y = clamp(WRIST.y, self.wrist_y_min, self.wrist_y_max)
    #         self.servo_angle[1] = map_range(wrist_y, self.wrist_y_min, self.wrist_y_max, self.y_max, self.y_min)
    #
    #         # calculate z angle
    #         palm_size = clamp(palm_size, self.plam_size_min, self.plam_size_max)
    #         self.servo_angle[2] = map_range(palm_size, self.plam_size_min, self.plam_size_max, self.z_max, self.z_min)
    #
    #         # float to int
    #         self.servo_angle = [int(i) for i in self.servo_angle]
    #     return self.servo_angle
    #
    # def rotate(self, frame):
    #     self.vertices, self.faces = self.__call__()
    #     print('Inside this')
    #     if write_video:
    #         fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #         out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))
    #
    #     mp_drawing = mp.solutions.drawing_utils
    #     mp_drawing_styles = mp.solutions.drawing_styles
    #     mp_hands = mp.solutions.hands
    #     with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    #         while cap.isOpened():
    #             success, image = cap.read()
    #             if not success:
    #                 print("Ignoring empty camera frame.")
    #                 # If loading a video, use 'break' instead of 'continue'.
    #                 continue
    #
    #             # To improve performance, optionally mark the image as not writeable to
    #             # pass by reference.
    #             image.flags.writeable = False
    #             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #             results = hands.process(image)
    #             # Draw the hand annotations on the image.
    #             image.flags.writeable = True
    #             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #             if results.multi_hand_landmarks:
    #                 print(len(results.multi_hand_landmarks))
    #
    #                 if len(results.multi_hand_landmarks) > 2:
    #                     if results.multi_hand_landmarks[0].landmark[0].x > results.multi_hand_landmarks[1].landmark[
    #                         0].x:
    #                         print("Left hand detected.")
    #                         hand_landmarks = results.multi_hand_landmarks[0]
    #                         self.servo_angle = self.landmark_to_servo_angle(hand_landmarks, 'left')
    #                     else:
    #                         print("Right hand detected.")
    #                         # print("One hand detected")
    #                         hand_landmarks = results.multi_hand_landmarks[0]
    #                         self.servo_angle = self.landmark_to_servo_angle(hand_landmarks, 'right')
    #                         if self.servo_angle != self.prev_servo_angle:
    #                             self.prev_servo_angle = self.servo_angle
    #                             servo_angle_str = str(self.prev_servo_angle)
    #                             formatted_str = servo_angle_str.replace(",", "")
    #                             print("Servo angle: ", formatted_str)
    #                             try:
    #                                 # self.client.send(bytes(formatted_str, 'UTF-8'))
    #                                 time.sleep(2)  # Pause execution for 1 second
    #                             except Exception as e:
    #                                 print("Error occurred during data sending:", e)
    #                 else:
    #                     print("More than two hands detected")
    #                 for hand_landmarks in results.multi_hand_landmarks:
    #                     mp_drawing.draw_landmarks(
    #                         image,
    #                         hand_landmarks,
    #                         mp_hands.HAND_CONNECTIONS,
    #                         mp_drawing_styles.get_default_hand_landmarks_style(),
    #                         mp_drawing_styles.get_default_hand_connections_style())
    #             # Flip the image horizontally for a selfie-view display.
    #             image = cv2.flip(image, 1)
    #             # show servo angle
    #             cv2.putText(image, str(self.servo_angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    #             cv2.imshow('MediaPipe Hands', image)
    #
    #             if write_video:
    #                 out.write(image)
    #             if cv2.waitKey(5) & 0xFF == 27:
    #                 if write_video:
    #                     out.release()
    #                 break
    #     cap.release()

        # self.client.close()
        # self.server.close()


    def __call__(self, enable_bundle_adjustment=False):
        pose_array = self.img_obj.K.ravel()
  #      print("Pose array shape:", pose_array.shape)
        transform_matrix_0 = np.eye(3, 4)
        transform_matrix_1 = np.empty((3, 4))

        pose_0 = np.dot(self.img_obj.K, transform_matrix_0)
        pose_1 = np.empty((3, 4))
        total_points = np.zeros((1, 3))
        total_colors = np.zeros((1, 3))

        image_0 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[0]))
        image_1 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[1]))

        feature_0, feature_1 = self.find_features(image_0, image_1)

        essential_matrix, em_mask = cv2.findEssentialMat(feature_0, feature_1, self.img_obj.K, method=cv2.RANSAC,prob=0.999, threshold=0.8, mask=None)
        feature_0 = feature_0[em_mask.ravel() == 1]
        feature_1 = feature_1[em_mask.ravel() == 1]

        _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(essential_matrix, feature_0, feature_1, self.img_obj.K)
        feature_0 = feature_0[em_mask.ravel() > 0]
        feature_1 = feature_1[em_mask.ravel() > 0]
        transform_matrix_1[:3, :3] = np.matmul(rot_matrix, transform_matrix_0[:3, :3])
        transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3],
                                                                          tran_matrix.ravel())

        pose_1 = np.dot(self.img_obj.K, transform_matrix_1)


        feature_0, feature_1, points_3d = self.triangulation(pose_0, pose_1, feature_0, feature_1)
        error, points_3d = self.reprojection_error(points_3d, feature_1, transform_matrix_1, self.img_obj.K,
                                                   homogenity=1)
        # Ideally error < 1
        _, _, feature_1, points_3d, _ = self.PnP(points_3d, feature_1, self.img_obj.K,
                                                 np.zeros((5, 1), dtype=np.float32),
                                                 feature_0, initial=1)

        total_images = len(self.img_obj.image_list) - 2
        pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))

        threshold = 0.8
        for i in tqdm(range(total_images), disable=True):
            image_2 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i + 2]))
            features_cur, features_2 = self.find_features(image_1, image_2)

            if i != 0:
                feature_0, feature_1, points_3d = self.triangulation(pose_0, pose_1, feature_0, feature_1)
                feature_1 = feature_1.T
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
                points_3d = points_3d[:, 0, :]
            cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = self.common_points(feature_1, features_cur, features_2)
            cm_points_2 = features_2[cm_points_1]
            cm_points_cur = features_cur[cm_points_1]

            rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = self.PnP(points_3d[cm_points_0],
                                                                                      cm_points_2,
                                                                                      self.img_obj.K,
                                                                                      np.zeros((5, 1),
                                                                                               dtype=np.float32),
                                                                                      cm_points_cur, initial=0)
            transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
            pose_2 = np.dot(self.img_obj.K, transform_matrix_1)

            error, points_3d = self.reprojection_error(points_3d, cm_points_2, transform_matrix_1, self.img_obj.K,
                                                       homogenity=0)

            cm_mask_0, cm_mask_1, points_3d = self.triangulation(pose_1, pose_2, cm_mask_0, cm_mask_1)
            error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K,
                                                       homogenity=1)

            pose_array = np.hstack((pose_array, pose_2.ravel()))

            if enable_bundle_adjustment:
                points_3d, cm_mask_1, transform_matrix_1 = self.bundle_adjustment(points_3d, cm_mask_1,
                                                                                  transform_matrix_1,
                                                                                  self.img_obj.K, threshold)
                pose_2 = np.dot(self.img_obj.K, transform_matrix_1)
                error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K,
                                                           homogenity=0)
      #          print("Bundle Adjusted error: ", error)
                total_points = np.vstack((total_points, points_3d))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left])
                total_colors = np.vstack((total_colors, color_vector))
            else:
                total_points = np.vstack((total_points, points_3d[:, 0, :]))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
                total_colors = np.vstack((total_colors, color_vector))

            transform_matrix_0 = np.copy(transform_matrix_1)
            pose_0 = np.copy(pose_1)
            image_0 = np.copy(image_1)
            image_1 = np.copy(image_2)
            feature_0 = np.copy(features_cur)
            feature_1 = np.copy(features_2)
            pose_1 = np.copy(pose_2)
            min_val = np.min(image_2)
            max_val = np.max(image_2)
  #      print(total_points.shape)
        self.vertices, self.faces = self.to_ply(self.img_obj.path, total_points, total_colors)
        np.savetxt(self.img_obj.path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2]+'_pose_array.csv', pose_array, delimiter = '\n')
        return self.vertices, self.faces

    #      print(self.vertices.shape)
    #      np.savetxt(self.img_obj.path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2]+'_pose_array.csv', pose_array, delimiter = '\n')

# def plot_and_display():
#     # self.ax.view_init(elev=0, azim=30)  # Adjust as needed
#     # self.fig.set_size_inches(15, 5)  # Width: 10 inches, Height: 5 inches
#     # # Adjust the spacing between subplots
#     # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)
#     fig, ax = plt.subplots()
#     ax.legend()
#     fig.canvas.draw()
#     image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#
#     # Convert RGB to BGR (OpenCV uses BGR format)
#     image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)
#     plt.close(fig)  # Close the figure after capturing the image
#
#     # Create a simple gradient image
#     image = np.linspace(0, 255, 300 * 300).reshape((300, 300)).astype(np.uint8)
#     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR format
#
#     # Define the canvas size and create a blank canvas filled with white (255)
#     height, width, channels = 600, 1000, 3
#     canvas = np.ones((height, width, channels), dtype=np.uint8) * 255
#
#     # Resize the images to fit the canvas layout
#     image_from_plot_resized = cv2.resize(image_from_plot, (int(width * 0.45), int(height * 0.8)))
#     image_resized = cv2.resize(image, (int(width * 0.4), int(height * 0.4)))
#
#     # Place the plot image on the left side of the canvas
#     canvas[int(height * 0.1):int(height * 0.9),
#     int(width * 0.05):int(width * 0.05) + image_from_plot_resized.shape[1]] = image_from_plot_resized
#
#     # Place the gradient image on the right side of the canvas
#     canvas[int(height * 0.1):int(height * 0.1) + image_resized.shape[0],
#     int(width * 0.55):int(width * 0.55) + image_resized.shape[1]] = image_resized
#
#     # Display the result using OpenCV
#     cv2.imshow('Combined Plot', canvas)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     self.ani = FuncAnimation(self.fg, self.rotate1, frames=20, interval=20, cache_frame_data=False)
#     plt.show()

    def init(self):
        credentials = self.authenticate()
        if credentials:
            print("Authentication successful!")
        else:
            print("Authentication failed. Check logs for details.")

        # Process images from Google Photos
        self.process_images(credentials)
        self.ax.view_init(elev=30, azim=45)  # Adjust as needed
        self.fig.set_size_inches(15, 5)  # Width: 10 inches, Height: 5 inches

        # Adjust the spacing between subplots
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)
        self.ani = FuncAnimation(self.fig, self.rotate, interval=100, cache_frame_data=False)
        plt.show()
        pass


sfm = Sfm("Datasets\\Many", 1,0,0,2)
sfm.init()