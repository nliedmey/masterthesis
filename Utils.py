import os
import subprocess
import shutil
import pandas as pd


def extractMp4():
    input_directory = '1_input_extract'
    output_directory = '1_output_extract'

    # Je MKV im 1_input_extract einen Ordner mit MKV+MP4 im Output erstellen
    for filename in os.listdir(input_directory):

        filename_input = os.path.join(input_directory, filename)

        # Output Folder erstellen
        filename_noMKV = filename[:-4]
        output_folder = os.path.join(output_directory, filename_noMKV)
        if not os.path.exists(output_folder):
            try:
                os.mkdir(output_folder)
            except OSError as error:
                print(error)
            filename_mp4 = filename_noMKV + '.mp4'
            filename_output = os.path.join(output_folder, filename_mp4)
            cmd_string = f"ffmpeg -i {filename_input} -map 0:0 -c copy -vsync vfr {filename_output}"

            # MP4 extrahieren und MKV in 1_output_extract moven
            try:
                proc = subprocess.Popen(cmd_string, shell=False)
                proc.wait()
            except:
                print("ERROR FFMPEG")

            # MKV-File aus Input in Output moven
            output_name = os.path.join(output_folder, filename)
            shutil.move(filename_input, output_name)
    return None


def parsXlsxToDf(path):
    # Pose Landmarks List
    pose_pose = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE',
                 'RIGHT_EYE_OUTER',
                 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
                 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
                 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
    frame = 0
    all_data = []

    coordinates_df = pd.read_excel(path, converters={"Frame": int})
    coordinates_df.rename(columns={'Unnamed: 0': 'Frame'}, inplace=True)
    while frame < len(coordinates_df):
        coordinate_lists_per_frame = []
        data_pose = {}
        for i in range(len(pose_pose)):
            coordinates_list = []
            # Wenn diese Landmarke in diesem Frame vorhanden ist...
            if coordinates_df.iloc[frame][pose_pose[i]]:
                # Koordinaten-Tupel (X,Y,Z,Vis) als String aus DF auslesen
                coordinates_string = (coordinates_df.iloc[frame][pose_pose[i]])
                # Unnötige Zeichen löschen
                coordinates_string = coordinates_string.replace("[", "")
                coordinates_string = coordinates_string.replace("]", "")
                # Liste aus String erstellen
                coordinates_list = list(map(float, coordinates_string.split(', ')))
                # Liste für diese Landmark an die Liste für den Frame anhängen
                coordinate_lists_per_frame.append(coordinates_list)
        # Koordinaten in DF sichern
        for j in range(len(pose_pose)):
            data_pose.update({pose_pose[j]: coordinate_lists_per_frame[j]}
                             )
        all_data.append(data_pose)
        frame += 1
    coordinates_df_final = pd.DataFrame(all_data)
    return coordinates_df_final


def getAllMp4(input_directory):
    list_mp4 = []
    for foldername in os.listdir(input_directory):
        for filename in os.listdir(input_directory + '/' + foldername):
            if filename.endswith('.mp4'):
                list_mp4.append(input_directory + '/' + foldername + '/' + filename)
    return list_mp4


def getAllMkv(input_directory):
    list_mkv = []
    for foldername in os.listdir(input_directory):
        for filename in os.listdir(input_directory + '/' + foldername):
            if filename.endswith('.mkv'):
                list_mkv.append(input_directory + '/' + foldername + '/' + filename)
    return list_mkv


def getAllXlsx(input_directory, select=None):
    list_xlsx = []
    for foldername in os.listdir(input_directory):
        if foldername == select:
            for filename in os.listdir(input_directory + '/' + foldername):
                if filename.endswith('.xlsx'):
                    list_xlsx.append(input_directory + '/' + foldername + '/' + filename)
    return list_xlsx
