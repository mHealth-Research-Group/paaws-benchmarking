# Pretrained Models from the PAAWS IMWUT `25 Paper Experiments

This folder contains the zipped weights of a random forest model (RF) that was trained to classify five activites (*Sitting*, *Standing*, *Walking*, *Biking*, *Lying_Down*) from left wrist (LeftWristTop) accelerometer data from 20 participants in the SimFL+Lab protocol. For more information on this model see [using the example model](#using-the-example-model).

The folder containing all the weights for the trained random forest models (RF) discussed in the benchmarking section of our paper (Sec. 4.2) can be downloaded from a <a href="https://drive.google.com/drive/folders/12Xr5isM4o_63GQXUstmpLAYKuu1uvIc9?usp=sharing" target="_blank">Google Drive folder</a> (~ 22 GB to download all zipped model weights, unzipped they are over 100 GB.) The rest of this README describes the folder structure of the models available for download on Google drive.

## Folder Structure

The weights for each model and their corresponding scaler (to be used on the pre-processed data prior to prediction) are in an idividual folder with the path name `/models/[training_data]/[num_activities]_Acts/[sensor]_[n]_participants/`. Possible values for each of these variables are defined below.

### [training_data]
`SimFL_Lab/`: folder containing models trained using accelerometer data from the SimFL+Lab protocol.

`FL/`: folder containing models trained using accelerometer data from the FL protocol.

### [num_activities]

`5_Acts/`: folder containing models trained to classify five activities: *Sitting*, *Standing*, *Walking*, *Biking*, *Lying_Down* (see `MAPPING_SCHEMES['lab_fl_5']` in `/utils.py` for more detail).

`9_Acts/`: folder containing models trained to classify nine activities: *Sitting*, *Standing*, *Walking*, *Biking*, *Lying_Down*, *Gym_Exercises*, *Household_Chores*, *Walking_Up_Stairs*, *Walking_Down_Stairs* (see `MAPPING_SCHEMES['lab_fl_9']` in `/utils.py` for more detail).

`11_Acts/`:  folder containing models trained to classify eleven activities: *Sitting*, *Standing*, *Walking*, *Biking*, *Lying_Down*, *Exercising*, *Household_Chores*, *Cooking*, *Driving*, *Grooming*, and *Eating/Drinking* (see `MAPPING_SCHEMES['fl_11']` in `/utils.py` for more detail).

`42_Acts/`: folder containing models trained to classify 42 different activities (see `MAPPING_SCHEMES['lab_42']` in `/utils.py` for more detail).

### [sensor]

`RightThigh`: models contained in a folder denoted `RightThigh` have been trained on accelerometer data obtained from the Right Thigh sensor.

`LeftWrist`: models contained in a folder denoted `LeftWrist` have been trained on accelerometer data obtained from the Left Wrist sensor (the LeftWristTop sensor in the case of SimFL+Lab data).

### [n]

`20`: models contained in a folder denoted `20` have been trained on accelerometer data from ~20 (18 in the case of FL) participants (see  `DATASETS['SimFL_20']` and `DATASETS['FL_20']` in `/utils.py` for more detail).

`126`: models contained in a folder denoted `126` have been trained on accelerometer data from 126 participants (see `DATASETS['126_3']` in `/utils.py` for more detail).

`252`: models contained in a folder denoted `252` have been trained on accelerometer data from 248 (some participants had missing data) participants (see `DATASETS['252']` in `/utils.py` for more detail).

### Using the Example Model

The weights of the model and scaler trained to classify five activities ([num_activities]=5) using left wrist ([sensor]=LeftWrist) data from the SimFL+Lab protocol ([training_data]=SimFL_Lab) from 20 participants ([n]=20) are in the folder: `/models/LeftWrist_5_Acts_20_Participants.zip`.

To use the weights from this model you must unzip this folder:
```bash
unzip models/SimFL_Lab/5_Activities/LeftWrist_20_Participants.zip -d models/SimFL_Lab/5_Activities/
```

The weights for the RF are in the file
`SimFL_LeftWrist_5_Acts_20_Participants_RF.joblib` and the scaler weights used to transform the data are in the file `SimFL_LeftWrist_5_Acts_20_Participants_SCALER.joblib`.

**NOTE**: In general, folders where models are stored are zipped for space. Each must be unzipped prior to using them. We recommend unzipping contents prior to using to ensure they work with our pathing in our code (line 170 in `/make_predictions.py`):
```bash
unzip [path_to_model_weights (include .zip file in path)] -d [path_to_model_weights_zip (do not include .zip file in path)]
```

## Example Usage

Each (unzipped) model can be run using the `/make_predictions.py` script by modifying the flags passed in on the command line.

To make predictions using the model trained to classify five activities using left wrist data from the SimFL+Lab protocol from 20 participants on DS_10's SimFL+Lab data (which *was included* in the original training) run
```bash
python make_predictions.py --ds_lo=10 --n=20 --sensor=LeftWrist --mapping="lab_fl_5" --out_file="predictions" --lab
```
The predictions will be output to a file in `/predictions/`.

**NOTE**: Because this data was included in the original training set, these results should not be used for analysis and this is for demonstration purposes only.