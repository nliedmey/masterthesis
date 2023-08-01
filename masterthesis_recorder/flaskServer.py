from flask import Flask
import signal
import os
import subprocess
import datetime
import csv
import json
import time
from pathlib import Path

app = Flask(__name__)
state = {}

@app.route("/start")
def start():
    probeId = "12345"
    fps = "30"
    depth = "NFOV_UNBINNED"
    color = "1080p"

    today = datetime.datetime.now()
    dateFormated = today.strftime("%Y-%m-%d")
    fullDateFormated = today.strftime("%Y%m%d%H%M%S%f")
    recorderPath = "C:\\Program Files\\Azure Kinect SDK v1.4.1\\tools\\k4arecorder.exe"

    folderPath = Path("F:\project\\storage\\") / dateFormated
    folderPath.mkdir(exist_ok=True, parents=True)

    # Create File-Name (Date_probeId_seqNr)
    fileName = fullDateFormated + "_" + str(probeId) + ".mkv"

    # Create CMD-String

    # cmdString = f"{recorderPath} -d {depth} -c {color} -r {fps} {folderPath}\\{fileName}"
    cmdList = [recorderPath, "-d", depth, "-c", color, "-r", fps, f"{str(folderPath/fileName)}"]
    metadata = {
        "fileName": fileName,
        "probeId": probeId,
        "depth": depth,
        "color": color,
        "fps": fps,
        "folderPath": str(folderPath)
    }
    print("Starting subprocess")
    state["proc"] = subprocess.Popen(cmdList, shell=False)
    print("Subprocess is running")
    state["metadata"] = metadata
    return "Started."


@app.route("/stop")
def stop():
    proc = state["proc"]
    metadata = state["metadata"]
    print(metadata)
    proc.send_signal(signal.CTRL_C_EVENT)
    stdout, stderr = proc.communicate()
    fileStats = os.stat(metadata["folderPath"] + "\\" + metadata["fileName"])
    sizeMB = round(fileStats.st_size / (1024 * 1024))
    metadata["sizeMB"] = sizeMB

    # Check whether header needs to be added or not
    fieldnames = ['fileName', 'probeId', 'depth', 'color', 'fps', 'folderPath', 'sizeMB']
    file_exists = os.path.isfile("F:\\project\\config\\conf.csv")
    # Write into conf.csv
    with open("F:\\project\\config\\conf.csv", mode='a', newline="") as confCSV:
        writer = csv.DictWriter(confCSV, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metadata)

    # Create json (not needed yet)
    jsonConf = json.dumps(metadata, indent=4)
    print(jsonConf)
    return "Stopped"

if __name__ == "__main__":
    def handler(signal, frame):
        print('CTRL-C pressed!')
    signal.signal(signal.SIGINT, handler)
    app.run()
