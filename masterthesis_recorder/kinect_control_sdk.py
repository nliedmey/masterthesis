import signal
import os
import shutil
import subprocess
import datetime
import csv
import json
from pathlib import Path


class KinectControlSDK:
    pid = 0
    fileSaved = False

    recorderPath = "C:\\Program Files\\Azure Kinect SDK v1.4.1\\tools\k4arecorder.exe"

    def __init__(self):
        self.proc = None
        self.metadata = {}

        self.record_filename = ""

        self.record_started = False

    # Record sequences
    def record_sequences(self, probeId, fps, depth, color):
        if self.record_started:
            return
        # Create Folder in D:\storage for today if not there yet
        today = datetime.datetime.now()
        dateFormated = today.strftime("%Y-%m-%d")
        fullDateFormated = today.strftime("%Y%m%d%H%M%S%f")

        folderPath = Path("C:\\VideoData\\project\\storage\\") / dateFormated
        folderPath.mkdir(exist_ok=True, parents=True)

        # Create File-Name (Date_probeId_seqNr)
        self.record_filename = fullDateFormated + "_" + str(probeId) + ".mkv"

        # Create CMD-String
        cmdString = self.recorderPath
        d = "-d " + str(depth)
        c = "-c " + str(color)
        r = "-r " + str(fps)
        ## time duration added currently
        cmdString = cmdString + " " + d + " " + c + " " + r + " " + str(folderPath) + "\\" + self.record_filename
        print(cmdString)

        self.metadata = {
            "fileName": self.record_filename,
            "probeId": probeId,
            "depth": depth,
            "color": color,
            "fps": fps,
            "folderPath": str(folderPath)
        }
        self.proc = subprocess.Popen(cmdString, shell=False)

        self.record_started = True

    # Stop sequences
    def stop_record(self, new_filename=None):
        if not self.record_started:
            return
        print("Signal send")
        self.proc.send_signal(signal.CTRL_C_EVENT)
        print("Signal sent")
        self.proc.wait()
        # Write used Kinect config to csv-file
        print("Writing config")
        self.write_config_csv()
        # rename record file when done
        if new_filename:
            self.rename_record(new_filename + ".mkv")
        self.record_started = False

    # Store used Kinect config parameters in csv file
    def write_config_csv(self):
        fileStats = os.stat(str(Path(self.metadata["folderPath"]) / self.metadata["fileName"]))
        sizeMB = round(fileStats.st_size / (1024 * 1024))
        self.metadata["sizeMB"] = sizeMB
        self.metadata["postProcessed"] = 0

        # Check whether header needs to be added or not
        fieldnames = ['fileName', 'probeId', 'depth', 'color', 'fps', 'folderPath', 'sizeMB', 'postProcessed']
        file_exists = os.path.isfile("C:\\VideoData\\project\\config\\conf.csv")
        # Write into conf.csv
        with open("C:\\VideoData\\project\\config\\conf.csv", mode='a', newline="") as confCSV:
            writer = csv.DictWriter(confCSV, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(self.metadata)

        # Create json (not needed yet)
        jsonConf = json.dumps(self.metadata, indent=4)
        print(jsonConf)

    def rename_record(self, new_name):
        dir_path = self.metadata.get("folderPath")
        file_path = os.path.join(dir_path, self.record_filename)
        new_path = os.path.join(dir_path, new_name)
        #os.rename(file_path, new_path)
        shutil.move(file_path, new_path)

