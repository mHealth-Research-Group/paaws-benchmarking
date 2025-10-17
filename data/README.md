# Sample Raw and Pre-Processed Data From the PAAWS IMWUT `25 Paper

So you can run our code and models without downloading the entire dataset, we provide some raw data (LeftWristTop data from DS 10 and DS 36 collected during the SimFL+Lab protocol). Additionally, we provide sample pre-processed data so you can verify the steps we use in pre-processing if you'd like to run our models on new datasets. To use the entirety of the PAAWS dataset with our code, we recommend placing the full dataset (e.g., PAAWS_SimFL_Lab or PAAWS_FreeLiving) in the `data/` folder. If the data is stored somewhere else, you may need to update the pathing in our code.

## Folder Structure

`feature_data/`: folder containing the features computed from the segmented left wrist top data from the SimFL+Lab protocol for DS_10 and DS_36. This data has been pre-processed and segmented to include five activities (see `MAPPING_SCHEMES['lab_fl_5']` in `/utils.py`).

`PAAWS_SimFL_Lab/`: folder containing **a small amount of sample** SimFL+Lab data collected from DS_10 and DS_36. Users of this dataset should import the full SimFL+Lab dataset into this folder to use the pathing already existing in our code.

**NOTE 1**: sample data is *all* collected from the LeftWristTop sensor worn during the the SimFL+Lab protocol by DS_10 and DS_36.

**NOTE 2**: the data in `feature_data/` is processed to include data from five activities (see `MAPPING_SCHEMES['lab_fl_5']` in `/utils.py`)

**NOTE 3**: sample data is stored in the same manner as the entire PAAWS dataset with accelerometer data in `accel/` and the corresponding annotations in `label/`.

