import subprocess

import ffmpeg
import numpy as np
import os
from pathlib import Path
import pandas as pd


class VideoProcessing:
    # TODO:
    #   1. Function to extract video tracks of all these clips
    def __init__(self):
        self.procMp4 = None
        self.procDepth = None
        self.procIR = None
        self.ffmpegPath = "C:\Program Files\\ffmpeg\\ffmpeg-master-latest-win64-gpl\\ffmpeg-master-latest-win64-gpl\\bin"
        self.listPaths = None
        self.dfClipsSelectedNew = None
        self.folderPathStorage = Path("G:\project\\storage\\")
        self.folderPathConfig = Path("G:\project\\config\\conf.csv")
        self.folderPathPostprocessed = Path("G:\project\\postprocessed")
        self.dfOldConf = pd.DataFrame()
        self.dfClipsSelected = pd.DataFrame()

    def getNonePostprocessedClips(self):
        # Read and store old config
        self.dfOldConf = pd.read_csv(self.folderPathConfig)
        # Select all none-postprocessed clips
        self.dfClipsSelected = self.dfOldConf.loc[self.dfOldConf['postProcessed'] == 0]
        # Store only the relevant coloumns for this class
        #self.dfClipsSelected = self.dfClipsSelected[["fileName", "folderPath"]]
        # print(self.dfClipsSelected)

    def createPostprocessedFolders(self):
        # Store unique date-folder paths
        self.listPaths = list(self.dfClipsSelected["folderPath"].unique())
        # For every of these folders...
        for path in self.listPaths:
            # create date-folders in postporcessed directory
            folderName = path[-10:]
            folderPathName = self.folderPathPostprocessed / folderName
            if not folderPathName.is_dir():
                folderPathName.mkdir(exist_ok=True, parents=True)
                print(str(folderPathName) + " created")

    def extractVideoTracks(self):
        # Append column with new path ("folderPathPostprocessed" / Datum / filename)
        self.dfClipsSelected = self.dfClipsSelected.assign(
            folderPathNew=lambda x: (self.folderPathPostprocessed / x['folderPath'].str[-10:] / x["fileName"].str[:-4]))
        # create a sub-directory for very clip if not existing yet
        listSubPaths = self.dfClipsSelected["folderPathNew"]
        for path in listSubPaths:
            if not path.is_dir():
                path.mkdir(exist_ok=True, parents=True)
                print("Done sub-dir")
        # For all available paths (Date-Folders) with not postprocessed files ...
        for pathX in self.listPaths:
            # Create a df with not postprocessed files of this path
            df = self.dfClipsSelected.loc[self.dfClipsSelected["folderPath"] == str(pathX)]
            # Create a list of the selected fileNames
            listFiles = df["fileName"].to_list()
            # Go through the date folder path
            for path, folder, files in os.walk(pathX):
                # For all files in this directory
                for file in files:
                    # If file is in list --> is not postprocessed
                    if file in listFiles:
                        print("Found file: %s" % file)
                        # Store the output folder for mp4, depth & IR files
                        output_folder = self.dfClipsSelected["folderPathNew"].loc[self.dfClipsSelected["fileName"] == file].values[0]
                        # Store name without extension and rename to XXX.mp4
                        name, ext = os.path.splitext(file)
                        out_name = name + ".mp4"
                        try:
                            #ToDO: Multi-Processing of ffmpeg commands
                            cmdStringMp4 = "ffmpeg -y -i " + str(pathX) + "\\" + str(
                                file) + " -map 0:0 -vsync 0 " + str(output_folder) + "\\" + str(
                                out_name) + " -map 0:1 -vsync 0 " + str(output_folder) + "\\" + "depth%04d.png"
                            self.procMp4 = subprocess.Popen(cmdStringMp4, shell=False)
                            self.procMp4.wait()
                        except:
                            # ffmpeg error will not be catched due to subprocessing...
                            print("Error video extraction")
                            self.dfClipsSelected = self.dfClipsSelected[self.dfClipsSelected.fileName != name]
                            print("Deleted: " + name)
                            pass
                    else:
                        pass
        # Change postProcessed value and write to config.csv
        print(self.dfClipsSelected)
        # ToDo: Now all clips get postProcess flag, even if extraction failed. Detect error in future
        self.dfClipsSelected['postProcessed'] = 1
        #print(self.dfClipsSelected)
        print("Ende Extract")

    def saveChangedConfCSV(self):
        print("Beginn SaveConf")
        # # change postProcessed values of processed files (dfClipsSelected) in old config (dfOldConf) and write to CSV file
        # listChangedFiles = self.dfClipsSelected["fileName"].to_list()
        # print(listChangedFiles)
        # print(self.dfOldConf)
        # self.dfOldConf = self.dfOldConf.loc[self.dfOldConf["fileName"].isin(listChangedFiles), 'postProcessed'] = 1
        # print(self.dfOldConf)
        try:
            self.dfClipsSelected.to_csv(self.folderPathConfig)
            print("try")
        except:
            print("csv store error")


    def convert_to_mp4(mkv_file, outputfolder):
        name, ext = os.path.splitext(mkv_file)
        out_name = name + ".mp4"
        ffmpeg.input(mkv_file).output(out_name).run()
        print("Finished converting {}".format(mkv_file))

    def startVideoPostprocess(self):
        self.getNonePostprocessedClips()
        self.createPostprocessedFolders()
        self.extractVideoTracks()
        self.saveChangedConfCSV()
