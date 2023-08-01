import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import time
import imutils
from helpers import colorize, convert_to_bgra_if_required
from pyk4a import PyK4APlayback
import pyk4a
from WorldCoordinates import calculateWorldTransMat
from Utils import getAllMp4, getAllMkv
import os
from statistics import median

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# Pose Landmarks List
pose_pose = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE',
             'RIGHT_EYE_OUTER',
             'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
             'LEFT_ELBOW',
             'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
             'RIGHT_INDEX',
             'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
             'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

pose_joints = [(15, 21), (16, 20), (18, 20), (3, 7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), (6, 8), (15, 17),
               (24, 26), (16, 22), (4, 5), (5, 6), (29, 31), (12, 24), (23, 24), (0, 1), (9, 10), (1, 2), (0, 4),
               (11, 13), (30, 32), (28, 32), (15, 19), (16, 18), (25, 27), (26, 28), (12, 14), (17, 19), (2, 3),
               (11, 12), (27, 29), (13, 15)]


def mediapipe_estimation(input_mp4=None, output_xlsx=None, verbose=False):
    # Create List of all .mp4-Files from 1_output_extract
    list_mp4 = getAllMp4(input_directory=input_mp4)
    # Load Mediapipe solution models
    mp_drawing = mp.solutions.drawing_utils
    # mp_holistic = mp.solutions.holistic
    mp_pose = mp.solutions.pose
    # Variables
    count = 0
    alldata = []
    alldata_world = []
    fps_time = 0
    no_landmark_counter = 0

    # Pose Landmarks
    pose_pose = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE',
                 'RIGHT_EYE_OUTER',
                 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
                 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST',
                 'LEFT_PINKY',
                 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
                 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
                 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

    # Start Capture Webcam or Replay
    # cap = cv2.VideoCapture("FINAL_DATA/PAD_Maske/20230509172241970_3_AOCEBGEZ_001/20230509172241970_3_AOCEBGEZ_001.mp4")
    # width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # print(fps)
    # Start mediapipe processing and cv2 actions
    with mp_pose.Pose(min_detection_confidence=0.75,
                      min_tracking_confidence=0.5, model_complexity=2, static_image_mode=False) as pose:
        for file in list_mp4:
            # Store id
            id = file[-36:-4]
            cap = cv2.VideoCapture(file)
            start = time.time()
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                # Flip image and convert to RGB for mediapipe processing
                # image = cv2.cvtColor(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_BGR2RGB)
                # image = imutils.resize(image, width=1920)

                # For original resolution
                # image = imutils.resize(image, width=1280)
                (height, width) = image.shape[:2]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Performance optimization for mediapipe processing
                image.flags.writeable = False
                # Process image with mediapipe holistic model to detect landmarks
                results = pose.process(image)
                # Undo writeable flag
                image.flags.writeable = True
                # convert back to BGR for CV2 functions
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # store original image for later imshow
                image_copy = np.copy(image)
                image_original = image

                # create new image with same shape but filled with zeros for landmarks
                image = np.zeros(image.shape)
                # draw pose landmarks on (empty image)
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # If pose landmarks are detected...
                data_pose = {}

                for i in range(len(pose_pose)):
                    if results.pose_landmarks:
                        pose_x = results.pose_landmarks.landmark[i].x * image.shape[1]
                        # Manchmal kommt es zu Pixel-Werten, die außerhalb der Auflösung liegen. Daher hier Korrektur der Werte auf einen Wert am Rand des Image
                        if pose_x > image.shape[1]:
                            if verbose:
                                print("Fehler Auflösung X")
                            pose_x = image.shape[1] - 1
                        if pose_x < 0:
                            if verbose:
                                print("Fehler Auflösung X")
                            pose_x = 1
                        pose_y = results.pose_landmarks.landmark[i].y * image.shape[0]
                        # Manchmal kommt es zu Pixel-Werten, die außerhalb der Auflösung liegen. Daher hier Korrektur der Werte auf einen Wert am Rand des Image
                        if pose_y > image.shape[0]:
                            if verbose:
                                print("Fehler Auflösung Y")
                            pose_y = image.shape[0] - 1
                        if pose_y < 0:
                            if verbose:
                                print("Fehler Auflösung Y")
                            pose_y = 1
                        pose_z = results.pose_landmarks.landmark[i].z
                        pose_visibility = results.pose_landmarks.landmark[i].visibility
                        pose_coords_combined = [pose_x, pose_y, pose_z, pose_visibility]

                        data_pose.update(
                            {pose_pose[i]: pose_coords_combined}
                        )

                    else:
                        # Wenn keine Landmarks detektiert wurden, sollen 0-Werte für X,Y,Z und visibility gespeichert werden
                        no_landmark_counter += 1
                        pose_coords_combined = [float(0), float(0), float(0), float(0)]
                        data_pose.update(
                            {pose_pose[i]: pose_coords_combined}
                        )

                alldata.append(data_pose)

                for i in range(len(pose_pose)):
                    cv2.circle(image_copy, (int(data_pose[pose_pose[i]][0]), int(data_pose[pose_pose[i]][1])), 5,
                               (0, 0, 255), -1)

                #image_copy = imutils.resize(image_copy, width=720)
                rotated_copy = cv2.rotate(image_copy, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if verbose:
                    cv2.imshow('Original Video', rotated_copy)
                    #cv2.imwrite("thesis_images/visFramesDots%d.jpg" % count, image_copy)
                    #cv2.imwrite("thesis_images/visFrames%d.jpg" % count, image_original)
                count += 1

                if cv2.waitKey(1) == ord('q'):
                    break
            df = pd.DataFrame(alldata)
            alldata = []
            end = time.time()
            duration = end - start
            df.to_excel(output_xlsx + '/' + id + ".xlsx")

            if verbose:
                print(f"Frames: {count}, No Landmarks detected: {no_landmark_counter}, Time elapsed: {duration}")
            cap.release()
            cv2.destroyAllWindows()
    return None


def depth_3d_estimation(input_mkv=None, input_xlsx=None, output_xlsx=None, interpolation=True, outlier_threshold=2, switch_axes=False,
                        verbose=False):
    list_mkv = getAllMkv(input_mkv)
    # open("transMat\\withAxisSwitch.txt", "w").close()
    # open("transMat\\withoutAxisSwitch.txt", "w").close()
    aruco_type = "DICT_5X5_250"

    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

    for mkv in list_mkv:
        id = mkv[-36:-4]
        playback_k4a = PyK4APlayback(mkv)
        playback_k4a.open()
        if verbose:
            print(playback_k4a.length)
            print(f"Record length: {playback_k4a.length / 1000000: 0.2f} sec")
        calibration = playback_k4a.calibration

        # Store camera parameters
        intrinsic_matrix = calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
        distortion_vector = calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR)
        # with open("thesis_images\\intrinsicMatrix.txt", "a") as myfile:
        #     myfile.write(str(intrinsic_matrix) + " \n")
        #     myfile.write(str(distortion_vector) + " \n")
        #     myfile.close()
        frame_nr = 0
        frame_nr_color = 0
        frame_nr_depth = 0

        # Load Landmark-Coordinates
        coordinates = pd.read_excel(input_xlsx + '/' + id + '.xlsx', converters={"Frame": int})
        coordinates.rename(columns={'Unnamed: 0': 'Frame'}, inplace=True)

        # Variables
        all_data = []
        coordinate_lists_per_frame = []
        coordinate_list_per_frame_final = []
        coords_processed = 0
        coords_3d_errors = 0
        coords_depth_0 = 0
        depth_3d_error = 0
        trans_mat_final = 0
        counter_120 = 0
        coords_depth_corrected = 0
        trans_mat_found = False
        start = time.time()


        while True:
            try:
                capture = playback_k4a.get_next_capture()
                color = capture.color
                depth = capture.depth

                if color is not None and depth is not None:

                    # 1. Versuch den Marker zu detektieren und die Transformationsmatrix aufzubauen
                    if trans_mat_found is False:
                        # with open("transMat\\withoutAxisSwitch.txt", "a") as myfile:
                        #     myfile.write("Intrinsic Matrix: ID " + id + " \n")
                        #     myfile.write(str(intrinsic_matrix) + "\n")
                        #     myfile.write("Distortion Vector: ID "+ id + " \n")
                        #     myfile.write(str(distortion_vector) + "\n")
                        #     myfile.close()
                        img = np.ascontiguousarray(color, dtype=np.uint8)
                        img = convert_to_bgra_if_required(playback_k4a.configuration["color_format"], img)
                        trans_mat_final = calculateWorldTransMat(frame=img, intrinsic_coefficients=intrinsic_matrix,
                                                                 distortion_coefficients=distortion_vector,
                                                                 aruco_dict_type=arucoDict, frame_number=frame_nr, switch_axis=switch_axes,
                                                                 id=id)
                        trans_mat_found = True
                    else:
                        trans_mat_final = np.load("transMat//transMat_default.npy")
                    for i in range(len(pose_pose)):
                        # Wenn diese Landmarke in diesem Frame vorhanden ist...
                        if coordinates.iloc[frame_nr][pose_pose[i]]:
                            # Koordinaten-Tupel (X,Y,Z,Vis) als String aus DF auslesen
                            coordinates_string = (coordinates.iloc[frame_nr][pose_pose[i]])
                            # Unnötige Zeichen löschen
                            coordinates_string = coordinates_string.replace("[", "")
                            coordinates_string = coordinates_string.replace("]", "")
                            # Liste aus String erstellen
                            coordinates_list = list(map(float, coordinates_string.split(', ')))
                            # print(type(coordinates_tuple))
                            # Liste für diese Landmark an die Liste für den Frame anhängen
                            coordinate_lists_per_frame.append(coordinates_list)
                    ## Display Streams
                    # Color Stream
                    # color_image = convert_to_bgra_if_required(playback_k4a.configuration["color_format"], color)
                    # cv2.imshow("k4a", color_image)
                    # cv2.imwrite("results/Depth/OHNE OPTIMIERUNGEN/frame_color%d.jpg" % frame_nr, color_image)

                    # Depth Stream
                    # cv2.imshow("grey", depth.astype(np.uint16))
                    # cv2.imshow("Depth", colorize(depth, (None, 5000)))
                    transformed = capture.transformed_depth
                    if verbose:
                        transformed_colorized = colorize(transformed, (None, 5000))
                    # transformed = imutils.resize(transformed, width=1920)
                    # Dict zum Speichern
                    data_pose = {}

                    if frame_nr > 0:
                        previous_frame_coords = all_data[frame_nr - 1]
                    # Zähler für die Spalte / Landmark
                    current_landmark = 0
                    # Für jedes Koordinaten List in diesem Frame...
                    for coord in coordinate_lists_per_frame:
                        interpolated = 0
                        this_lm_depth_error = False
                        this_lm_3d_error = False
                        # Wenn X,Y nicht 0 sind... (Also Landmarks gefunden wurden)
                        if coord[0] != 0 and coord[1] != 0:
                            # Ermittle den Tiefenwert (in mm)
                            # Bei images zuerst y-Wert und dann x-Wert, siehe hier (https://github.com/etiennedub/pyk4a/issues/161)
                            depth_value = transformed[int(coord[1])][int(coord[0])]
                            if depth_value == 120:
                                counter_120 += 1
                            if depth_value == 0:
                                this_lm_depth_error = True
                                coords_depth_0 += 1
                            # Tiefenwert an Koordinaten hängen
                            coord.append(depth_value)
                            # Umwandlung von 2D-Koordinatensystem (X und Y in Pixel + Z in millimeter) in 3D-Koordinatensystem der Kamera
                            # mit Übergabe der 2D-Koordinaten
                            coords_processed += 1
                            try:
                                coord_3d_system = calibration.convert_2d_to_3d(
                                    coordinates=(int(coord[0]), int(coord[1])),
                                    depth=depth_value,
                                    source_camera=pyk4a.CalibrationType.COLOR)
                                # 3D X und Y Werte ergänzen
                                x_3d = coord_3d_system[0]
                                y_3d = coord_3d_system[1]
                                coord.append(x_3d)
                                coord.append(y_3d)

                            except ValueError:
                                # Wenn convert nicht funktioniert hat, ergänze [0,0]
                                coords_3d_errors += 1
                                this_lm_3d_error = True
                                if verbose:
                                    print("---- RGB 3D Probleme ----")
                                    print(
                                        f"frame: {frame_nr} landmark: {pose_pose[current_landmark]} x:  {int(coord[0])} y: {int(coord[1])} depth: {depth_value}")
                                # TEST 3D- und Weltkoordianten auf vorherigem Frame nehmen
                                x_3d = 0
                                y_3d = 0
                                coord.extend([0, 0])
                            # 3D Weltkoordinaten berechnen
                            camera_vector = np.array([[x_3d], [y_3d], [depth_value], [1]])
                            # if verbose:
                            #     print("---- CAMERA VECTOR -----")
                            #     print(camera_vector)
                            #     print("---- TRANS MAT FINAL -----")
                            #     print(trans_mat_final)
                            # If transformation matrix is found in the frames, take it. Else, take the default one

                            try:
                                world_coordinates = trans_mat_final.dot(camera_vector)
                            except AttributeError:
                                #### Default trans_mat aus anderem Video laden
                                trans_mat_final = np.load("transMat//transMat_default.npy")
                            # print("----- WORLD COORDINATES ----")
                            # print(world_coordinates)

                            # Welt Koordinaten ergänzen
                            x_world = world_coordinates[0][0]
                            y_world = world_coordinates[1][0]
                            z_world = world_coordinates[2][0]
                            if x_world != 0 and y_world != 0 and z_world != 0:
                                coord.append(x_world)
                                coord.append(y_world)
                                coord.append(z_world)
                            else:
                                x_world = previous_frame_coords[pose_pose[current_landmark]][7]
                                y_world = previous_frame_coords[pose_pose[current_landmark]][8]
                                z_world = previous_frame_coords[pose_pose[current_landmark]][9]
                                coord.append(x_world)
                                coord.append(y_world)
                                coord.append(z_world)
                                interpolated += 1

                        # Wenn keine Koordinate vorhanden ist...
                        else:
                            print("In keine Koordinate gefunden")
                            depth_value = 0
                            # Ergänze Tiefenwert, x_3d, y_3d als 0,0,0
                            coord.extend([0, 0, 0])
                        # Prüfen ob Depth und 3D error gemeinsam vorkommen
                        if this_lm_3d_error == True and this_lm_depth_error == True:
                            depth_3d_error += 1
                        # colorized = colorize(transformed, (None, 5000))
                        if verbose:
                            cv2.circle(transformed_colorized, (int(coord[0]), int(coord[1])), 10, (0, 0, 0), -1)

                        # cv2.circle(depth, (int(coord[0]), int(coord[1])), 5, (120, 200, 50), -1)
                        coord.append(interpolated)
                        coordinate_list_per_frame_final.append(coord)
                        # Zähler hoch
                        current_landmark += 1

                    # cv2.imwrite("snapshots/transformed_colorized_frame%d.jpg" % frame_nr,
                    #             transformed_colorized)

                    # Koordinaten in DF sichern
                    for i in range(len(pose_pose)):
                        data_pose.update(
                            {pose_pose[i]: coordinate_list_per_frame_final[i]}
                        )
                    # previous_coords = data_pose
                    # print("---- PREVIOUS COORDS")
                    # print(previous_coords)
                    all_data.append(data_pose)

                    # Anzeige des Transformed Bild. Obere Variante ist farblich, aber kann Flackern erzeugen
                    if verbose:
                        # transformed_copy = imutils.resize(transformed, width=720)
                        rotated_transformed_copy = cv2.rotate(transformed_colorized, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        cv2.imshow("Transformed Depth Colorized", rotated_transformed_copy)
                    # depth_copy = imutils.resize(depth, width=720)
                    # colorized_depth = colorize(depth, (None, 5000))
                    # colorized = colorize(rotated_transformed_copy, (None, 5000))
                    # cv2.imwrite("snapshots/transformed_colorized_frame%d.jpg" % frame_nr,
                    #             colorize(rotated_transformed_copy, (None, 5000)))
                    # cv2.imwrite("thesis_images/depth_basis_grayframe%d.jpg" % frame_nr, depth)
                    # cv2.imwrite("thesis_images/colorized_depth%d.jpg" % frame_nr, colorized_depth)
                    # cv2.imwrite("thesis_images/transformed%d.jpg" % frame_nr, colorize(transformed, (None, 5000)))
                    frame_nr += 1
                else:
                    if depth is None:
                        "depth none"
                    if color is None:
                        "color none"
                # Wenn alle Koordinaten abgearbeitet wurden, dann raus aus der Schleife
                coordinate_lists_per_frame = []
                coordinate_list_per_frame_final = []
                if frame_nr == coordinates.shape[0]:
                    break
                key = cv2.waitKey(1)
                if key != -1:
                    break
            except EOFError:
                break
        cv2.destroyAllWindows()
        playback_k4a.close()
        # Neues Dataframe mit allen Landmark-Koordinaten inkl Tiefenwerte und 3D-Koordinaten erstellen und abspeichern
        df = pd.DataFrame(all_data)

        ######################## INTERPOLATION ###########################
        if interpolation:
            # First Missing/Invalid Interpolation
            df = missingInvalidInterpolation(df=df, verbose=verbose)
            # Second Joint outliers
            df = anatomicalInterpolation(df=df, outlier_threshold=outlier_threshold, verbose=verbose)
            df.to_excel(output_xlsx + '/' + id + '_3d_interpolated.xlsx')
        else:
            df.to_excel(output_xlsx + '/' + id + '_3d.xlsx')
        # df.to_json("coordinates_updated.json")
        end = time.time()
        duration = end - start
        if verbose:
            print(
                f"Total Frames: {frame_nr}, Color Frames: {frame_nr_color}, Depth Frames: {frame_nr_depth}, Coords processed "
                f"{coords_processed}, 3D coords error {coords_3d_errors}, Empty depth {coords_depth_0}, "
                f"Depth corrected {coords_depth_corrected}, Empty Depth + 3D Error zeitgleich {depth_3d_error}, "
                f"Time elapsed: {duration}, 120 Depth: {counter_120}")


def depth_3d_estimation_evaluation(input_mkv=None, input_xlsx=None, output_xlsx=None, interpolation=True, outlier_threshold=2,
                        verbose=False):
    list_mkv = getAllMkv(input_mkv)
    for mkv in list_mkv:
        id = mkv[-36:-4]
        playback_k4a = PyK4APlayback(mkv)
        playback_k4a.open()
        if verbose:
            print(playback_k4a.length)
            print(f"Record length: {playback_k4a.length / 1000000: 0.2f} sec")
        calibration = playback_k4a.calibration

        # Store camera parameters
        intrinsic_matrix = calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
        distortion_vector = calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR)

        frame_nr = 0
        frame_nr_color = 0
        frame_nr_depth = 0

        # Load Landmark-Coordinates
        coordinates = pd.read_excel(input_xlsx + '/' + id + '.xlsx', converters={"Frame": int})
        coordinates.rename(columns={'Unnamed: 0': 'Frame'}, inplace=True)

        # Variables
        all_data = []
        coordinate_lists_per_frame = []
        coordinate_list_per_frame_final = []
        coords_processed = 0
        coords_3d_errors = 0
        coords_depth_0 = 0
        depth_3d_error = 0
        trans_mat_final = 0
        counter_120 = 0
        coords_depth_corrected = 0
        trans_mat_found = False
        start = time.time()

        aruco_type = "DICT_5X5_250"

        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

        while True:
            try:
                capture = playback_k4a.get_next_capture()
                color = capture.color
                depth = capture.depth

                if color is not None and depth is not None:
                    #### Default trans_mat aus anderem Video laden
                    trans_mat_final = np.load("transMat//transMat_default.npy")
                    for i in range(len(pose_pose)):
                        # Wenn diese Landmarke in diesem Frame vorhanden ist...
                        if coordinates.iloc[frame_nr][pose_pose[i]]:
                            # Koordinaten-Tupel (X,Y,Z,Vis) als String aus DF auslesen
                            coordinates_string = (coordinates.iloc[frame_nr][pose_pose[i]])
                            # Unnötige Zeichen löschen
                            coordinates_string = coordinates_string.replace("[", "")
                            coordinates_string = coordinates_string.replace("]", "")
                            # Liste aus String erstellen
                            coordinates_list = list(map(float, coordinates_string.split(', ')))
                            # print(type(coordinates_tuple))
                            # Liste für diese Landmark an die Liste für den Frame anhängen
                            coordinate_lists_per_frame.append(coordinates_list)
                    ## Display Streams
                    # Color Stream
                    # color_image = convert_to_bgra_if_required(playback_k4a.configuration["color_format"], color)
                    # cv2.imshow("k4a", color_image)
                    # cv2.imwrite("results/Depth/OHNE OPTIMIERUNGEN/frame_color%d.jpg" % frame_nr, color_image)

                    # Depth Stream
                    # cv2.imshow("grey", depth.astype(np.uint16))
                    # cv2.imshow("Depth", colorize(depth, (None, 5000)))
                    transformed = capture.transformed_depth
                    # transformed = imutils.resize(transformed, width=1920)
                    # Dict zum Speichern
                    data_pose = {}

                    if frame_nr > 0:
                        previous_frame_coords = all_data[frame_nr - 1]
                    # Zähler für die Spalte / Landmark
                    current_landmark = 0
                    # Für jedes Koordinaten List in diesem Frame...
                    for coord in coordinate_lists_per_frame:
                        interpolated = 0
                        this_lm_depth_error = False
                        this_lm_3d_error = False
                        # Wenn X,Y nicht 0 sind... (Also Landmarks gefunden wurden)
                        if coord[0] != 0 and coord[1] != 0:
                            # Ermittle den Tiefenwert (in mm)
                            depth_value = transformed[int(coord[1])][int(coord[0])]
                            if depth_value == 120:
                                counter_120 += 1
                            if depth_value == 0:
                                this_lm_depth_error = True
                                coords_depth_0 += 1
                            # Tiefenwert an Koordinaten hängen
                            coord.append(depth_value)
                            # Umwandlung von 2D-Koordinatensystem (X und Y in Pixel + Z in millimeter) in 3D-Koordinatensystem der Kamera
                            # mit Übergabe der 2D-Koordinaten
                            coords_processed += 1
                            try:
                                coord_3d_system = calibration.convert_2d_to_3d(
                                    coordinates=(int(coord[0]), int(coord[1])),
                                    depth=depth_value,
                                    source_camera=pyk4a.CalibrationType.COLOR)
                                # 3D X und Y Werte ergänzen
                                x_3d = coord_3d_system[0]
                                y_3d = coord_3d_system[1]
                                coord.append(x_3d)
                                coord.append(y_3d)

                            except ValueError:
                                # Wenn convert nicht funktioniert hat, ergänze [0,0]
                                coords_3d_errors += 1
                                this_lm_3d_error = True
                                if verbose:
                                    print("---- RGB 3D Probleme ----")
                                    print(
                                        f"frame: {frame_nr} landmark: {pose_pose[current_landmark]} x:  {int(coord[1])} y: {int(coord[0])} depth: {depth_value}")
                                # TEST 3D- und Weltkoordianten auf vorherigem Frame nehmen
                                x_3d = 0
                                y_3d = 0
                                coord.extend([0, 0])
                            # 3D Weltkoordinaten berechnen
                            camera_vector = np.array([[x_3d], [y_3d], [depth_value], [1]])
                            # if verbose:
                            #     print("---- CAMERA VECTOR -----")
                            #     print(camera_vector)
                            #     print("---- TRANS MAT FINAL -----")
                            #     print(trans_mat_final)
                            world_coordinates = trans_mat_final.dot(camera_vector)
                            # print("----- WORLD COORDINATES ----")
                            # print(world_coordinates)

                            # Welt Koordinaten ergänzen
                            x_world = world_coordinates[0][0]
                            y_world = world_coordinates[1][0]
                            z_world = world_coordinates[2][0]
                            if x_world != 0 and y_world != 0 and z_world != 0:
                                coord.append(x_world)
                                coord.append(y_world)
                                coord.append(z_world)
                            else:
                                x_world = previous_frame_coords[pose_pose[current_landmark]][7]
                                y_world = previous_frame_coords[pose_pose[current_landmark]][8]
                                z_world = previous_frame_coords[pose_pose[current_landmark]][9]
                                coord.append(x_world)
                                coord.append(y_world)
                                coord.append(z_world)
                                interpolated += 1

                        # Wenn keine Koordinate vorhanden ist...
                        else:
                            print("In keine Koordinate gefunden")
                            depth_value = 0
                            # Ergänze Tiefenwert, x_3d, y_3d als 0,0,0
                            coord.extend([0, 0, 0])
                        # Prüfen ob Depth und 3D error gemeinsam vorkommen
                        if this_lm_3d_error == True and this_lm_depth_error == True:
                            depth_3d_error += 1
                        cv2.circle(transformed, (int(coord[0]), int(coord[1])), 10, (120, 0, 50), -1)
                        # cv2.circle(depth, (int(coord[0]), int(coord[1])), 5, (120, 200, 50), -1)
                        coord.append(interpolated)
                        coordinate_list_per_frame_final.append(coord)
                        # Zähler hoch
                        current_landmark += 1

                    # Koordinaten in DF sichern
                    for i in range(len(pose_pose)):
                        data_pose.update(
                            {pose_pose[i]: coordinate_list_per_frame_final[i]}
                        )
                    # previous_coords = data_pose
                    # print("---- PREVIOUS COORDS")
                    # print(previous_coords)
                    all_data.append(data_pose)

                    # Anzeige des Transformed Bild. Obere Variante ist farblich, aber kann Flackern erzeugen
                    if verbose:
                        transformed_copy = imutils.resize(transformed, width=720)
                        rotated_transformed_copy = cv2.rotate(transformed_copy, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        cv2.imshow("Transformed Depth Colorized", colorize(rotated_transformed_copy, (None, 5000)))
                    # depth_copy = imutils.resize(depth, width=720)
                    # colorized_depth = colorize(depth, (None, 5000))
                    # colorized = colorize(transformed, (None, 5000))
                    # cv2.imwrite("snapshots/transformed_colorized_frame%d.jpg" % frame_nr,
                    #             colorize(transformed_copy, (None, 5000)))
                    # cv2.imwrite("results/OHNE OPTIMIERUNGEN/depthframes/depth_basis_grayframe%d.jpg" % frame_nr, depth)
                    frame_nr += 1
                else:
                    if depth is None:
                        "depth none"
                    if color is None:
                        "color none"
                # Wenn alle Koordinaten abgearbeitet wurden, dann raus aus der Schleife
                coordinate_lists_per_frame = []
                coordinate_list_per_frame_final = []
                if frame_nr == coordinates.shape[0]:
                    break
                key = cv2.waitKey(1)
                if key != -1:
                    break
            except EOFError:
                break
        cv2.destroyAllWindows()
        playback_k4a.close()
        # Neues Dataframe mit allen Landmark-Koordinaten inkl Tiefenwerte und 3D-Koordinaten erstellen und abspeichern
        df = pd.DataFrame(all_data)

        ######################## INTERPOLATION ###########################
        if interpolation:
            # First Missing/Invalid Interpolation
            df = missingInvalidInterpolation(df=df, verbose=verbose)
            # Second Joint outliers
            df = anatomicalInterpolation(df=df, outlier_threshold=outlier_threshold, verbose=verbose)
            df.to_excel(output_xlsx + '/' + id + '_3d_interpolated.xlsx')
        else:
            df.to_excel(output_xlsx + '/' + id + '_3d.xlsx')
        # df.to_json("coordinates_updated.json")
        end = time.time()
        duration = end - start
        if verbose:
            print(
                f"Total Frames: {frame_nr}, Color Frames: {frame_nr_color}, Depth Frames: {frame_nr_depth}, Coords processed "
                f"{coords_processed}, 3D coords error {coords_3d_errors}, Empty depth {coords_depth_0}, "
                f"Depth corrected {coords_depth_corrected}, Empty Depth + 3D Error zeitgleich {depth_3d_error}, "
                f"Time elapsed: {duration}, 120 Depth: {counter_120}")


def missingInvalidInterpolation(df, verbose=False):
    """
    This function interpolates missing or invalid depth and 3d values by replacing them with the last known valid
    value for this landmark.

    :param verbose: Print additional Infos
    :param df: DataFrame with all coordinates
    :return: df: with interpolated depth and 3d values
    """
    counter = 0
    counter_3d = 0
    corrected = 0
    corrected_3d = 0
    i = 0
    while i < len(df):
        for j in range(len(pose_pose)):
            # Depth-Value == 0 or == 120
            depth = df.iloc[i][pose_pose[j]][4]
            if depth == 0 or depth == 120:
                counter += 1
                # Not in first frame
                if i > 0:
                    # Current depth & 3d coords = previous depth & 3d coords
                    df.iloc[i][pose_pose[j]][4] = df.iloc[i - 1][pose_pose[j]][4]
                    df.iloc[i][pose_pose[j]][5] = df.iloc[i - 1][pose_pose[j]][5]
                    df.iloc[i][pose_pose[j]][6] = df.iloc[i - 1][pose_pose[j]][6]
                    df.iloc[i][pose_pose[j]][7] = df.iloc[i - 1][pose_pose[j]][7]
                    df.iloc[i][pose_pose[j]][8] = df.iloc[i - 1][pose_pose[j]][8]
                    df.iloc[i][pose_pose[j]][9] = df.iloc[i - 1][pose_pose[j]][9]
                    corrected += 1
                    # Set interpolated flag
                    df.iloc[i][pose_pose[j]][10] = 1
            # Invalid 3d values
            array_3d = df.iloc[i][pose_pose[j]][5:7]
            if 0 in array_3d:
                counter_3d += 1
                # Not in first frame
                if i > 0:
                    # Take previous camera and world 3d coordinates
                    df.iloc[i][pose_pose][j][5:10] = df.iloc[i - 1][pose_pose[j]][5:10]
                    corrected_3d += 1
                    # Set interpolated flag
                    df.iloc[i][pose_pose[j]][10] = 1

        i += 1
    if verbose:
        print(f"Counter: {counter}, Corrected: {corrected}, Counter_3d: {counter_3d}, Corrected_3d: {corrected_3d}")
    return df


def anatomicalInterpolation(df, outlier_threshold=2, verbose=False):
    # joints should have stable length in a 3d space
    # due to landmarks at the slight edge of the human body, some depth measurements are incorrect
    # the depth is then measured at the ground floor which lead to wrong joints
    list_names = []
    list_median = []
    i = 0
    all_data = []
    for j in range(len(pose_joints)):
        coord1_lmName = pose_pose[pose_joints[j][0]]
        coord2_lmName = pose_pose[pose_joints[j][1]]
        joint_name = "joint_" + coord1_lmName + "_" + coord2_lmName
        list_names.append(joint_name)
    while i < len(df):
        list_distance = []
        coord_dict = {}
        for j in range(len(pose_joints)):
            coord1_lmName = pose_pose[pose_joints[j][0]]
            coord2_lmName = pose_pose[pose_joints[j][1]]
            coord1 = df.iloc[i][coord1_lmName][4:7]
            coord2 = df.iloc[i][coord2_lmName][4:7]
            euclDistance = np.linalg.norm(np.array(coord1) - np.array(coord2))
            list_distance.append(euclDistance)
        for n in range(len(list_names)):
            coord_dict.update(
                {list_names[n]: list_distance[n]}
            )
        all_data.append(coord_dict)
        i += 1
    df_joints = pd.DataFrame(all_data)
    ## All joint distances are stored in df_joints

    ## Next: Look for outliers with X-% above or below median distance of a joint
    i = 0
    for col in list_names:
        median_col = median(df_joints[col])
        list_median.append(median_col)
    while i < len(df_joints):
        for j in range(len(pose_joints)):
            current_dist = df_joints.iloc[i][j]
            current_median = list_median[j]
            upper_threshold = current_median + current_median * outlier_threshold
            lower_threshold = current_median - current_median * outlier_threshold
            if current_dist > upper_threshold or current_dist < lower_threshold:
                if verbose:
                    print("OUTLIER")
                ## INTERPOLATE ANATOMICAL OUTLIER HERE
                # Plan:
                # - Outlier Joints are known now
                # - A joint consists of two landmarks
                # - You need to identify which of the both landmarks produce the outlier error
                # - Approach: Take both landmarks and compare them to their previous coordinate
                #               - First Try: Only depth comparison because this is the jumping value
                # Not for first frame
                no_error = 0
                if i > 0:
                    coord1_lmName = pose_pose[pose_joints[j][0]]
                    coord2_lmName = pose_pose[pose_joints[j][1]]
                    current_depth_lm1 = df.iloc[i][coord1_lmName][4]
                    current_depth_lm2 = df.iloc[i][coord2_lmName][4]
                    previous_depth_lm1 = df.iloc[i - 1][coord1_lmName][4]
                    previous_depth_lm2 = df.iloc[i - 1][coord2_lmName][4]
                    if verbose:
                        print(df.iloc[i][coord1_lmName][4:7])
                        print(df.iloc[i - 1][coord1_lmName][4:7])
                        print(df.iloc[i][coord2_lmName][4:7])
                        print(df.iloc[i - 1][coord2_lmName][4:7])
                    # Comparison of current landmark depth with previous to detect the jumping landmark
                    if current_depth_lm1 > (previous_depth_lm1 * 1.5) or current_depth_lm1 < (previous_depth_lm1 * 0.5):
                        if verbose:
                            print(
                                f"LM1 depth error, frame {i}, lm {coord1_lmName},current dist {current_dist}, upper "
                                f"threshold {upper_threshold} , lower threshold {lower_threshold}")
                        df.iloc[i][coord1_lmName][4] = df.iloc[i - 1][coord1_lmName][4]
                        df.iloc[i][coord1_lmName][5] = df.iloc[i - 1][coord1_lmName][5]
                        df.iloc[i][coord1_lmName][6] = df.iloc[i - 1][coord1_lmName][6]
                        df.iloc[i][coord1_lmName][7] = df.iloc[i - 1][coord1_lmName][7]
                        df.iloc[i][coord1_lmName][8] = df.iloc[i - 1][coord1_lmName][8]
                        df.iloc[i][coord1_lmName][9] = df.iloc[i - 1][coord1_lmName][9]
                        # Set interpolation flag to 1
                        df.iloc[i][coord1_lmName][10] = 1
                    else:
                        no_error = 1
                    if current_depth_lm2 > (previous_depth_lm2 * 1.5) or current_depth_lm2 < (previous_depth_lm2 * 0.5):
                        if verbose:
                            print(
                            f"LM2 depth error, frame {i}, lm {coord2_lmName}, current dist {current_dist}, upper "
                            f"threshold {upper_threshold} , lower threshold {lower_threshold}")
                        df.iloc[i][coord2_lmName][4] = df.iloc[i - 1][coord2_lmName][4]
                        df.iloc[i][coord2_lmName][5] = df.iloc[i - 1][coord2_lmName][5]
                        df.iloc[i][coord2_lmName][6] = df.iloc[i - 1][coord2_lmName][6]
                        df.iloc[i][coord2_lmName][7] = df.iloc[i - 1][coord2_lmName][7]
                        df.iloc[i][coord2_lmName][8] = df.iloc[i - 1][coord2_lmName][8]
                        df.iloc[i][coord2_lmName][9] = df.iloc[i - 1][coord2_lmName][9]
                        # Set interpolation flag to 1
                        df.iloc[i][coord2_lmName][10] = 1
                    else:
                        no_error = 1
                    if no_error == 1:
                        if verbose:
                            print(
                            f"NO DEPTH ERROR, frame {i}, with current dist {current_dist}, upper threshold {upper_threshold} , lower threshold {lower_threshold}")
                        ## Beide Landmark Coordintes auf ihren vorherigen Wert setzen
                        df.iloc[i][coord1_lmName][4] = df.iloc[i - 1][coord1_lmName][4]
                        df.iloc[i][coord1_lmName][5] = df.iloc[i - 1][coord1_lmName][5]
                        df.iloc[i][coord1_lmName][6] = df.iloc[i - 1][coord1_lmName][6]
                        df.iloc[i][coord1_lmName][7] = df.iloc[i - 1][coord1_lmName][7]
                        df.iloc[i][coord1_lmName][8] = df.iloc[i - 1][coord1_lmName][8]
                        df.iloc[i][coord1_lmName][9] = df.iloc[i - 1][coord1_lmName][9]
                        df.iloc[i][coord2_lmName][4] = df.iloc[i - 1][coord2_lmName][4]
                        df.iloc[i][coord2_lmName][5] = df.iloc[i - 1][coord2_lmName][5]
                        df.iloc[i][coord2_lmName][6] = df.iloc[i - 1][coord2_lmName][6]
                        df.iloc[i][coord2_lmName][7] = df.iloc[i - 1][coord2_lmName][7]
                        df.iloc[i][coord2_lmName][8] = df.iloc[i - 1][coord2_lmName][8]
                        df.iloc[i][coord2_lmName][9] = df.iloc[i - 1][coord2_lmName][9]
                        # Set interpolation flag to 1
                        df.iloc[i][coord1_lmName][10] = 1
            else:
                if verbose:
                    print("NO OUTLIER")
        i += 1
    return df


def depth_3d_estimation_liveMissingInvalidInterpolation(input_mkv=None, input_xlsx=None, output_xlsx=None,
                                                        verbose=False):
    list_mkv = getAllMkv(input_mkv)
    for mkv in list_mkv:
        id = mkv[-36:-4]
        playback_k4a = PyK4APlayback(mkv)
        playback_k4a.open()
        if verbose:
            print(playback_k4a.length)
            print(f"Record length: {playback_k4a.length / 1000000: 0.2f} sec")
        calibration = playback_k4a.calibration

        # Store camera parameters
        intrinsic_matrix = calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
        distortion_vector = calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR)
        # print(intrinsic_matrix)
        # print(distortion_vector)

        frame_nr = 0
        frame_nr_color = 0
        frame_nr_depth = 0

        # Pose Landmarks List
        pose_pose = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE',
                     'RIGHT_EYE_OUTER',
                     'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
                     'LEFT_ELBOW',
                     'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
                     'RIGHT_INDEX',
                     'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                     'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
        # Landmark Koordinaten laden
        coordinates = pd.DataFrame()
        coordinates = pd.read_excel(input_xlsx + '/' + id + '.xlsx', converters={"Frame": int})
        coordinates.rename(columns={'Unnamed: 0': 'Frame'}, inplace=True)
        all_data = []
        coordinates_list = []
        coordinate_lists_per_frame = []
        coordinate_list_per_frame_final = []
        coordinates_list_previous = []
        coords_processed = 0
        coords_3d_errors = 0
        coords_depth_0 = 0
        depth_3d_error = 0
        world_error = 0
        trans_mat_final = 0
        counter_120 = 0
        coords_depth_corrected = 0
        coords_3d_corrected = 0
        interpolated = 0
        this_lm_3d_error = False
        this_lm_depth_error = False
        color_available = False
        trans_mat_found = False
        start = time.time()

        aruco_type = "DICT_5X5_250"

        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

        while True:
            try:
                # Capture Objekt (Frame) erstellen
                capture = playback_k4a.get_next_capture()
                color = capture.color
                depth = capture.depth

                if color is not None and depth is not None:

                    # 1. Versuch den Marker zu detektieren und die Transformationsmatrix aufzubauen
                    if trans_mat_found is False:
                        img = np.ascontiguousarray(color, dtype=np.uint8)
                        img = convert_to_bgra_if_required(playback_k4a.configuration["color_format"], img)
                        # img = imutils.resize(img, width=1920)
                        trans_mat_final = calculateWorldTransMat(frame=img, intrinsic_coefficients=intrinsic_matrix,
                                                                 distortion_coefficients=distortion_vector,
                                                                 aruco_dict_type=arucoDict, frame_number=frame_nr)
                        trans_mat_found = True
                    # für alle Landmarken (33)
                    for i in range(len(pose_pose)):
                        coordinates_list = []
                        # Wenn diese Landmarke in diesem Frame vorhanden ist...
                        if coordinates.iloc[frame_nr][pose_pose[i]]:
                            # print(coordinates.iloc[frame_nr][pose_pose[i]])
                            # TODO: Gibt derzeit nur String zurück --> DERZEIT konvert String to list
                            # Koordinaten-Tupel (X,Y,Z,Vis) als String aus DF auslesen
                            coordinates_string = (coordinates.iloc[frame_nr][pose_pose[i]])
                            # Unnötige Zeichen löschen
                            coordinates_string = coordinates_string.replace("[", "")
                            coordinates_string = coordinates_string.replace("]", "")
                            # Liste aus String erstellen
                            coordinates_list = list(map(float, coordinates_string.split(', ')))
                            # print(type(coordinates_tuple))
                            # Liste für diese Landmark an die Liste für den Frame anhängen
                            coordinate_lists_per_frame.append(coordinates_list)

                    ## Display Streams
                    # Color Stream
                    # color_image = convert_to_bgra_if_required(playback_k4a.configuration["color_format"], color)
                    # cv2.imshow("k4a", color_image)
                    # cv2.imwrite("results/Depth/OHNE OPTIMIERUNGEN/frame_color%d.jpg" % frame_nr, color_image)

                    # Depth Stream
                    # cv2.imshow("grey", depth.astype(np.uint16))
                    # cv2.imshow("Depth", colorize(depth, (None, 5000)))
                    transformed = capture.transformed_depth
                    # transformed = imutils.resize(transformed, width=1920)
                    # Dict zum Speichern
                    data_pose = {}

                    if frame_nr > 0:
                        previous_frame_coords = all_data[frame_nr - 1]
                    # Zähler für die Spalte / Landmark
                    current_landmark = 0
                    # ToDo: Interpolation von nicht erkannten Tiefenwerten.
                    # Für jedes Koordinaten List in diesem Frame...
                    for coord in coordinate_lists_per_frame:
                        interpolated = 0
                        this_lm_depth_error = False
                        this_lm_3d_error = False
                        # Wenn X,Y nicht 0 sind... (Also Landmarks gefunden wurden)
                        if coord[0] != 0 and coord[1] != 0:
                            # Ermittle den Tiefenwert (in mm)
                            depth_value = transformed[int(coord[1])][int(coord[0])]
                            if depth_value == 120:
                                counter_120 += 1
                                if frame_nr > 0:
                                    coords_depth_corrected += 1
                                    previous_frame_depth = previous_frame_coords[pose_pose[current_landmark]][4]
                                    depth_value = previous_frame_depth
                                    interpolated += 1
                            if depth_value == 0:
                                this_lm_depth_error = True
                                coords_depth_0 += 1

                                ## Zuletzt bekannten Tiefenwert der Landmark nehmen (ab 2. Frame)
                                if frame_nr > 0:
                                    coords_depth_corrected += 1
                                    previous_frame_depth = previous_frame_coords[pose_pose[current_landmark]][4]
                                    depth_value = previous_frame_depth
                                    interpolated += 1
                            # Tiefenwert an Koordinaten hängen
                            coord.append(depth_value)
                            # Umwandlung von 2D-Koordinatensystem (X und Y in Pixel + Z in millimeter) in 3D-Koordinatensystem der Kamera
                            # mit Übergabe der 2D-Koordinaten
                            coords_processed += 1
                            try:
                                coord_3d_system = calibration.convert_2d_to_3d(
                                    coordinates=(int(coord[1]), int(coord[0])),
                                    depth=depth_value,
                                    source_camera=pyk4a.CalibrationType.COLOR)
                                # 3D X und Y Werte ergänzen
                                x_3d = coord_3d_system[0]
                                y_3d = coord_3d_system[1]
                                coord.append(x_3d)
                                coord.append(y_3d)

                            except ValueError:
                                # Wenn convert nicht funktioniert hat, ergänze [0,0]
                                coords_3d_errors += 1
                                this_lm_3d_error = True
                                if verbose:
                                    print("---- RGB 3D Probleme ----")
                                    print(
                                        f"frame: {frame_nr} landmark: {pose_pose[current_landmark]} x:  {int(coord[1])} y: {int(coord[0])} depth: {depth_value}")
                                ## TEST 3D- und Weltkoordianten auf vorherigem Frame nehmen
                                if frame_nr > 0:
                                    coords_3d_x_from_previous = previous_frame_coords[pose_pose[current_landmark]][5]
                                    coords_3d_y_from_previous = previous_frame_coords[pose_pose[current_landmark]][6]
                                    # Ergänze x_3d, y_3d als 0,0
                                    coord.extend([coords_3d_x_from_previous, coords_3d_y_from_previous])
                                    interpolated += 1
                                else:
                                    coord.extend([0, 0])
                            # 3D Weltkoordinaten berechnen
                            camera_vector = np.array([[x_3d], [y_3d], [depth_value], [1]])
                            if verbose:
                                print("---- CAMERA VECTOR -----")
                                print(camera_vector)
                                print("---- TRANS MAT FINAL -----")
                                print(trans_mat_final)
                            world_coordinates = trans_mat_final.dot(camera_vector)

                            # Welt Koordinaten ergänzen
                            x_world = world_coordinates[0][0]
                            y_world = world_coordinates[1][0]
                            z_world = world_coordinates[2][0]
                            if x_world != 0 and y_world != 0 and z_world != 0:
                                coord.append(x_world)
                                coord.append(y_world)
                                coord.append(z_world)
                            else:
                                world_error += 1
                                x_world = previous_frame_coords[pose_pose[current_landmark]][7]
                                y_world = previous_frame_coords[pose_pose[current_landmark]][8]
                                z_world = previous_frame_coords[pose_pose[current_landmark]][9]
                                coord.append(x_world)
                                coord.append(y_world)
                                coord.append(z_world)
                                interpolated += 1

                        # Wenn keine Koordinate vorhanden ist...
                        else:
                            if verbose:
                                print("In keine Koordinate gefunden")
                            depth_value = 0
                            # Ergänze Tiefenwert, x_3d, y_3d als 0,0,0
                            coord.extend([0, 0, 0])
                        # Prüfen ob Depth und 3D error gemeinsam vorkommen
                        if this_lm_3d_error == True and this_lm_depth_error == True:
                            depth_3d_error += 1
                        cv2.circle(transformed, (int(coord[0]), int(coord[1])), 10, (120, 0, 50), -1)
                        # cv2.circle(depth, (int(coord[0]), int(coord[1])), 5, (120, 200, 50), -1)
                        coord.append(interpolated)
                        coordinate_list_per_frame_final.append(coord)
                        # Zähler hoch
                        current_landmark += 1

                    # Koordinaten in DF sichern
                    for i in range(len(pose_pose)):
                        data_pose.update(
                            {pose_pose[i]: coordinate_list_per_frame_final[i]}
                        )
                    # previous_coords = data_pose
                    # print("---- PREVIOUS COORDS")
                    # print(previous_coords)
                    all_data.append(data_pose)

                    # Anzeige des Transformed Bild. Obere Variante ist farblich, aber kann Flackern erzeugen
                    if verbose:
                        transformed_copy = imutils.resize(transformed, width=720)
                        rotated_transformed_copy = cv2.rotate(transformed_copy, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        cv2.imshow("Transformed Depth Colorized", colorize(rotated_transformed_copy, (None, 5000)))
                    # depth_copy = imutils.resize(depth, width=720)
                    # colorized_depth = colorize(depth, (None, 5000))
                    # colorized = colorize(transformed, (None, 5000))
                    # cv2.imwrite("snapshots/transformed_colorized_frame%d.jpg" % frame_nr,
                    #             colorize(transformed_copy, (None, 5000)))
                    # cv2.imwrite("results/OHNE OPTIMIERUNGEN/depthframes/depth_basis_grayframe%d.jpg" % frame_nr, depth)
                    frame_nr += 1
                else:
                    if depth is None:
                        "depth none"
                    if color is None:
                        "color none"
                # Wenn alle Koordinaten abgearbeitet wurden, dann raus aus der Schleife
                coordinate_lists_per_frame = []
                coordinate_list_per_frame_final = []
                if frame_nr == coordinates.shape[0]:
                    break
                key = cv2.waitKey(1)
                if key != -1:
                    break
            except EOFError:
                break
        cv2.destroyAllWindows()
        playback_k4a.close()
        # Neues Dataframe mit allen Landmark-Koordinaten inkl Tiefenwerte und 3D-Koordinaten erstellen und abspeichern
        df = pd.DataFrame(all_data)
        df.to_excel(output_xlsx + '/' + id + '_3d_interpolated.xlsx')
        # df.to_json("coordinates_updated.json")
        end = time.time()
        duration = end - start
        if verbose:
            print(
                f"Total Frames: {frame_nr}, Color Frames: {frame_nr_color}, Depth Frames: {frame_nr_depth}, "
                f"Coords processed {coords_processed}, 3D coords error {coords_3d_errors}, "
                f"Empty depth {coords_depth_0}, Depth corrected {coords_depth_corrected}, "
                f"Empty Depth + 3D Error zeitgleich {depth_3d_error}, Time elapsed: {duration}, "
                f"120 Depth: {counter_120}, world_error: {world_error}")
