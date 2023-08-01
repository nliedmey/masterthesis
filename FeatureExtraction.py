from Utils import parsXlsxToDf
import numpy as np
from scipy.ndimage import uniform_filter1d
import pandas as pd


def euclideanDistance3dCalc(lm1, lm2, xlsxFile=None, onlyDistSeriesReturn=False, coordSystem=None, df_in=None,
                            outlierCorrection=False):
    """
    This function computes the euclidean distance in a 3D-Space for the selected landmarks.

    :param lm1: Landmark 1 of the euclidean distance
    :param lm2: Landmark 2 of the euclidean distance
    :param xlsxFile: The 1_input_extract xlsx-Files with coordinates
    :param onlyDistSeriesReturn: If True, returns the distance as a Series. By default, it returns the 1_input_extract coordinates with an attached distance column
    :param coordSystem: By default, the World-3D coordinates are selected. If set to "kinect", the RGB-3D-Space coordinates are selected
    :param df_in: If coordinates already loaded in a DataFrame, put it here and xlsx get ignored
    :param outlierCorrection: If True, the outlier in the distance get smoothed
    :return: Returns either a coordinate dataframe with distance column attached or only distance as data series
    """
    if coordSystem == 'kinect':
        # Camera 3D Coordinate System
        startIndex = 4
        endIndex = 7
    else:
        # World 3D Coordinate System
        startIndex = 7
        endIndex = 10

    if df_in is None:
        df = parsXlsxToDf(xlsxFile)
    else:
        df = df_in

    lm1coords = np.array([x[startIndex:endIndex] for x in df[lm1]])
    lm2coords = np.array([x[startIndex:endIndex] for x in df[lm2]])
    euclDistance = np.linalg.norm(lm1coords - lm2coords, axis=1)

    if outlierCorrection:
        # Outlier glätten
        euclDistance = uniform_filter1d(euclDistance, size=10)

    if onlyDistSeriesReturn:
        return np.array(euclDistance)

    colName = lm1 + '_' + lm2 + '_euclDist'
    df[colName] = euclDistance
    print("Done")
    return df


def euclideanDistance3dCalcToOrigin(lm1, xlsxFile=None, onlyDistSeriesReturn=False, coordSystem=None, df_in=None,
                                    outlierCorrection=False):
    """
    This function computes the euclidean distance in a 3D-Space for the selected landmarks and the coordinate system origin.

    :param lm1: Landmark 1 of the euclidean distance
    :param xlsxFile: The 1_input_extract xlsx-Files with coordinates
    :param onlyDistSeriesReturn: If True, returns the distance as a Series. By default, it returns the 1_input_extract coordinates with an attached distance column
    :param coordSystem: By default, the World-3D coordinates are selected. If set to "kinect", the RGB-3D-Space coordinates are selected
    :param df_in: If coordinates already loaded in a DataFrame, put it here and xlsx get ignored
    :param outlierCorrection: If True, the outlier in the distance get smoothed
    :return: Returns either a coordinate dataframe with distance column attached or only distance as data series
    """
    if coordSystem == 'kinect':
        # Camera 3D Coordinate System
        startIndex = 4
        endIndex = 7
    else:
        # World 3D Coordinate System
        startIndex = 7
        endIndex = 10

    if df_in is None:
        df = parsXlsxToDf(xlsxFile)
    else:
        df = df_in

    lm1coords = np.array([x[startIndex:endIndex] for x in df[lm1]])
    lm2coords = [0, 0, 0]
    euclDistance = np.linalg.norm(lm2coords - lm1coords, axis=1)
    if outlierCorrection:
        # Outlier glätten
        euclDistance = uniform_filter1d(euclDistance, size=10)

    if onlyDistSeriesReturn:
        return np.array(euclDistance)

    colName = lm1 + '_origin_euclDist'
    df[colName] = euclDistance
    print("Done")
    return df


def angleThreePoints3dCalc(lm1, lm2, lm3, df_in=None, xlsxFile=None, coordSystem=None, onlyAngleSeriesReturn=False):
    """
    This function computes the angle between three landmarks in a 3D-space.
    :param lm1:
    :param lm2:
    :param lm3:
    :param df_in:
    :param xlsxFile:
    :param coordSystem:
    :param onlyAngleSeriesReturn:
    :return:
    """
    if coordSystem == 'kinect':
        # Camera 3D Coordinate System
        startIndex = 4
        endIndex = 7
    else:
        # World 3D Coordinate System
        startIndex = 7
        endIndex = 10

    lm1coords = np.array([x[startIndex:endIndex] for x in df_in[lm1]])
    lm2coords = np.array([x[startIndex:endIndex] for x in df_in[lm2]])
    lm3coords = np.array([x[startIndex:endIndex] for x in df_in[lm3]])
    lm2lm1 = lm1coords - lm2coords
    lm2lm3 = lm3coords - lm2coords

    cosine_numerator = np.sum(lm2lm1 * lm2lm3, axis=1)
    cosine_denominator_1 = np.linalg.norm(lm2lm1, axis=1)
    cosine_denominator_2 = np.linalg.norm(lm2lm3, axis=1)
    cosine_angle = cosine_numerator / (cosine_denominator_1 * cosine_denominator_2)
    angles = np.arccos(cosine_angle)
    degree_angles = np.rad2deg(angles)

    return degree_angles


def calculate_angle_between_vectors(line1_lm1, line1_lm2, line2_lm1, line2_lm2, df_in=None):
    """
    This function calculates the angle between two lines in a 3D-space using the provided landmarks.

    :param line1_lm1: Landmark 1 of line 1
    :param line1_lm2: Landmark 2 of line 1
    :param line2_lm1: Landmark 1 of line 2
    :param line2_lm2: Landmark 2 of line 2
    :param df_in: If coordinates are already loaded in a DataFrame, put it here
    :return: Returns a Series with the angles between the lines
    """
    if df_in is None:
        raise ValueError("A DataFrame with coordinates must be provided.")

    # For World coordinate system
    startIndex = 7
    endIndex = 10

    lm1coords = np.array([x[startIndex:endIndex] for x in df_in[line1_lm1]])
    lm2coords = np.array([x[startIndex:endIndex] for x in df_in[line1_lm2]])
    lm3coords = np.array([x[startIndex:endIndex] for x in df_in[line2_lm1]])
    lm4coords = np.array([x[startIndex:endIndex] for x in df_in[line2_lm2]])

    # Calculate the vectors between the end points of each line
    vector1 = lm2coords - lm1coords
    vector2 = lm4coords - lm3coords

    # Calculate the angle between the vectors
    dot_product = np.sum(vector1 * vector2, axis=1)
    norm_product = np.linalg.norm(vector1, axis=1) * np.linalg.norm(vector2, axis=1)
    cosine_angle = dot_product / norm_product
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_head_rotation(lm1, lm2, lm3, df_in=None):
    """
    This function calculates the head rotation using Euler angles based on the provided landmarks.

    :param lm1: Landmark 1 of the head (e.g., nose)
    :param lm2: Landmark 2 of the head (e.g., left ear)
    :param lm3: Landmark 3 of the head (e.g., right ear)
    :param df_in: If coordinates are already loaded in a DataFrame, put it here
    :return: Returns the head rotation as Euler angles
    """
    if df_in is None:
        raise ValueError("A DataFrame with coordinates must be provided.")

    startIndex = 7
    endIndex = 10

    lm1_coords = np.array([x[startIndex:endIndex] for x in df_in[lm1]])
    lm2_coords = np.array([x[startIndex:endIndex] for x in df_in[lm2]])
    lm3_coords = np.array([x[startIndex:endIndex] for x in df_in[lm3]])

    # Calculate the vectors between the landmarks
    vector1 = lm2_coords - lm1_coords  # Vector from landmark 1 to landmark 2
    vector2 = lm3_coords - lm1_coords  # Vector from landmark 1 to landmark 3

    # Calculate the cross product to get the rotation axis
    rotation_axis = np.cross(vector1, vector2)
    rotation_axis /= np.linalg.norm(rotation_axis, axis=1, keepdims=True)

    # Calculate the rotation angles using arctan2
    rotation_angles = np.degrees(np.arctan2(rotation_axis[:, 1], rotation_axis[:, 0]))

    return rotation_angles


def calculate_head_rotation_euler(lm1, lm2, lm3, df_in=None):
    """
    This function calculates the head rotation using Euler angles based on the provided landmark names and a DataFrame.

    :param lm1: Name of the first landmark column in the DataFrame
    :param lm2: Name of the second landmark column in the DataFrame
    :param lm3: Name of the third landmark column in the DataFrame
    :param df_in: DataFrame containing the landmark coordinates
    :return: Returns the head rotation as Euler angles (yaw, pitch, roll)
    """
    if df_in is None:
        raise ValueError("A DataFrame with coordinates must be provided.")

    startIndex = 7
    endIndex = 10

    lm1_coords = np.array([x[startIndex:endIndex] for x in df_in[lm1].values])
    lm2_coords = np.array([x[startIndex:endIndex] for x in df_in[lm2].values])
    lm3_coords = np.array([x[startIndex:endIndex] for x in df_in[lm3].values])

    # Convert lists to NumPy arrays
    lm1_coords = np.array(lm1_coords)
    lm2_coords = np.array(lm2_coords)
    lm3_coords = np.array(lm3_coords)

    # Calculate the vectors between the landmarks
    vector1 = lm2_coords - lm1_coords
    vector2 = lm3_coords - lm1_coords

    # Calculate the yaw angle
    yaw_angle = np.degrees(np.arctan2(vector1[:, 1], vector1[:, 0]))

    # Calculate the pitch angle
    reference_vector = np.array([0, 0, 1])  # Reference vector for pitch calculation (e.g., straight ahead)
    projection_vector = np.dot(vector1, reference_vector)[:, np.newaxis] * reference_vector
    projection_norm = np.linalg.norm(projection_vector, axis=1)
    pitch_angle = np.degrees(np.arcsin(projection_norm / np.linalg.norm(vector1, axis=1)))

    # Calculate the roll angle
    roll_angle = np.degrees(np.arctan2(vector2[:, 2], vector2[:, 0]))

    return yaw_angle, pitch_angle, roll_angle

def calculate_head_rotation_euler2(df_in):

    startIndex = 7
    endIndex = 10

    nose = np.array([x[startIndex:endIndex] for x in df_in["NOSE"].values])
    left_eye = np.array([x[startIndex:endIndex] for x in df_in["LEFT_EYE"].values])
    right_eye = np.array([x[startIndex:endIndex] for x in df_in["RIGHT_EYE"].values])

    # Calculate the vector between the eyes
    eye_vector = right_eye - left_eye

    # Calculate the vector between the nose and the midpoint between the eyes
    nose_to_eye_midpoint = (left_eye + right_eye) / 2 - nose

    # Calculate the yaw angle
    yaw = np.arctan2(eye_vector[1], eye_vector[0])

    # Calculate the pitch angle
    pitch = np.arctan2(nose_to_eye_midpoint[2], np.linalg.norm(nose_to_eye_midpoint[:2]))

    # Calculate the roll angle
    roll = np.arctan2(nose_to_eye_midpoint[1], nose_to_eye_midpoint[0])

    # Convert angles from radians to degrees
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    roll_deg = np.degrees(roll)

    return yaw_deg, pitch_deg, roll_deg
