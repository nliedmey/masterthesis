# Master Thesis - Automated Supervision for Biometric Applications through 3D Human Pose Analysis

## Installation
- Azure Kinect SDK from https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md
- FFMPEG (Instruction: https://cran.r-project.org/web/packages/act/vignettes/install_ffmpeg.html)
- requirement.txt

## Execution
- "FINAL-NOTEBOOK_ArtifactPrototype"
  - Artifact Prototype for 3D Pose Data Generation
  - Place .mkv-Record in folder "1_input_extract" and run the Notebook cells
- "FINAL-NOTEBOOK_UseCase1"
  - Visualization of the generated 3D pose data
  - If needed, adapt the .xlsx-Filepaths
- "FINAL-NOTEBOOK_UseCase2"
  - Classification of BonaFide, Mask-Attack and Shirt-Attack
  - Exemplary records for training / prediction are stored under "4_input_classification"
