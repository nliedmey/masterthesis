import cv2
import numpy as np


def calculateWorldTransMat(frame, intrinsic_coefficients, distortion_coefficients, aruco_dict_type, id, frame_number, switch_axis=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = cv2.aruco.DetectorParameters_create()
    trans_mat = 0
    marker_length = 161
    # Für diesen Frame, detektiere alle Marker und speichere ihre Eckpunkte und IDs ab
    total_corners, total_ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict_type,
                                                                            parameters=parameters)
    # Wenn Eckpunkte, also Marker vorhanden sind
    if len(total_corners) > 0:
        # Berechne rotations und translation vectors für alle Marker
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(total_corners, marker_length, intrinsic_coefficients,
                                                                       distortion_coefficients)
        # cv2.drawFrameAxes(frame, intrinsic_coefficients, distortion_coefficients, rvec, tvec, 2500, 5)
        # cv2.putText(frame, f"id: {id[0]} Distance: {round(distance, 3)}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        # if frame_number < 10:
        #         cv2.imwrite("snapshots/coordinatesystem{}.jpg".format(frame_number), frame)
        ## CREATE TRANSFORMATION MATRIX FROM ROTATION AND TRANSLATION VECTORS
        # convert rvec to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        if switch_axis:
            # Erstelle eine zusätzliche Rotationsmatrix für die Koordinatensystem-Anpassung
            # z-Achse invertieren
            adjustment_rmat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
            # Wende die Anpassungs-Rotationsmatrix an
            rmat = np.dot(rmat, adjustment_rmat)
        # create a 4x4 transformation matrix with translation vector
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = rmat
        trans_mat[:3, 3] = tvec.ravel()
    # with open("transMat\\withoutAxisSwitch.txt", "a") as myfile:
    #     myfile.write("rvec: ID " + id + " \n")
    #     myfile.write(str(rvec) + "\n")
    #     myfile.write("tvec: ID " + id + " \n")
    #     myfile.write(str(tvec) + "\n")
    #     myfile.write("rmat: ID " + id + " \n")
    #     myfile.write(str(rmat) + "\n")
    #     myfile.write("trans_mat: ID " + id + " \n")
    #     myfile.write(str(trans_mat) + "\n")
    #     myfile.close()
    np.save(f'transMat_{id}.npy', trans_mat)
    return trans_mat


def calculateWorldTransMatWithCameraAngle(frame, intrinsic_coefficients, distortion_coefficients, aruco_dict_type, id, frame_number, switch_axis=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = cv2.aruco.DetectorParameters_create()
    trans_mat = 0
    marker_length = 161
    camera_orientation_angles = [0, 0, 180]
    # Für diesen Frame, detektiere alle Marker und speichere ihre Eckpunkte und IDs ab
    total_corners, total_ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict_type,
                                                                            parameters=parameters)
    # Wenn Eckpunkte, also Marker vorhanden sind
    if len(total_corners) > 0:
        # Berechne rotations und translation vectors für alle Marker
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(total_corners, marker_length, intrinsic_coefficients,
                                                                       distortion_coefficients)
        cv2.drawFrameAxes(frame, intrinsic_coefficients, distortion_coefficients, rvec, tvec, 3000, 3)
        # cv2.putText(frame, f"id: {id[0]} Distance: {round(distance, 3)}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        # cv2.imwrite("originals/screenshots/test_koordinatensystem{}.jpg".format(frame_number), frame)
        ## CREATE TRANSFORMATION MATRIX FROM ROTATION AND TRANSLATION VECTORS
        # convert rvec to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        if switch_axis:
            # Erstelle eine zusätzliche Rotationsmatrix für die Koordinatensystem-Anpassung
            # z-Achse invertieren
            # adjustment_rmat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
            # Apply camera orientation adjustments around X, Y, and Z axes
            adjustment_rmat = np.array([[1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, -1]])
            rvec = np.dot(rvec, adjustment_rmat)
            rmat, _ = cv2.Rodrigues(rvec)

            camera_orientation_rmat_x = np.array([[1, 0, 0],
                                                  [0, np.cos(np.radians(camera_orientation_angles[0])),
                                                   -np.sin(np.radians(camera_orientation_angles[0]))],
                                                  [0, np.sin(np.radians(camera_orientation_angles[0])),
                                                   np.cos(np.radians(camera_orientation_angles[0]))]])

            camera_orientation_rmat_y = np.array([[np.cos(np.radians(camera_orientation_angles[1])), 0,
                                                   np.sin(np.radians(camera_orientation_angles[1]))],
                                                  [0, 1, 0],
                                                  [-np.sin(np.radians(camera_orientation_angles[1])), 0,
                                                   np.cos(np.radians(camera_orientation_angles[1]))]])

            camera_orientation_rmat_z = np.array([[np.cos(np.radians(camera_orientation_angles[2])),
                                                   -np.sin(np.radians(camera_orientation_angles[2])), 0],
                                                  [np.sin(np.radians(camera_orientation_angles[2])),
                                                   np.cos(np.radians(camera_orientation_angles[2])), 0],
                                                  [0, 0, 1]])
            rmat = np.dot(rmat, camera_orientation_rmat_x)
            rmat = np.dot(rmat, camera_orientation_rmat_y)
            rmat = np.dot(rmat, camera_orientation_rmat_z)
            # Wende die Anpassungs-Rotationsmatrix an
            # rmat = np.dot(rmat, adjustment_rmat)
        # create a 4x4 transformation matrix with translation vector
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = rmat
        trans_mat[:3, 3] = tvec.ravel()
    # with open("transMat\\withoutAxisSwitch.txt", "a") as myfile:
    #     myfile.write("rvec: ID " + id + " \n")
    #     myfile.write(str(rvec) + "\n")
    #     myfile.write("tvec: ID " + id + " \n")
    #     myfile.write(str(tvec) + "\n")
    #     myfile.write("rmat: ID " + id + " \n")
    #     myfile.write(str(rmat) + "\n")
    #     myfile.write("trans_mat: ID " + id + " \n")
    #     myfile.write(str(trans_mat) + "\n")
    #     myfile.close()
    np.save(f'transMat_{id}.npy', trans_mat)
    return trans_mat


def aruco_display(image, corners, ids, rejected):
    if len(corners) > 0:

        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))
    return image
