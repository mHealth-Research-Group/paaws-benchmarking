# Raw Results from the PAAWS IMWUT `25 Paper

We release the raw predictions from all the experiments described in Sec. 4.2: Benchmarking the PAAWS R1 Dataset. Additionally, we release the aggregate confusion matrices (CMs) for our experiments and larger versions of our in-paper figures. Lastly, we release a preprint of our paper.

## Folder Structure

`predictions/`: folder containing the raw predictions resulting from leave-one-participant-out cross validation (LOPO CV) during benchmarking.

*For the SimFL+Lab and FL benchmarking (Sec 4.2.1 and Sec 4.2.2)* predictions from each model during training and evaluation are stored in subfolders with the following naming convention: `results/predictions/[sec_num_title]/[data_used_in_validation]_Validation/[num_activities]_Activities/[sensor]_[n]_Participants/[sec_num]_[validation_data]_Validation_[num_activities]_Activities_[sensor]_[n]_Participants_DS_[ds_lo].csv`.

*For PSG label benchmarking (Sec 4.2.3)* results are stored in subfolders with the naming convention: `results/predictions/[sec_num_title]/[sensor]/[sec_num]_[sensor]_DS_[ds_lo].csv`.

Each variable is explained in the [path variables section](#path-variables).

Each of these prediction CSVs contains the timestamp the window started (col: `"START_TIME"`), the true label (col: `"MAPPED_LABEL"` or `"PSG_TRUE"`), and the prediction (col: `"PREDICTION"`).

**NOTE**: folders have been zipped at the "[sec_num_title]" level to reduce the size of this repository. Please unzip each file for the raw results.

`confusion_matrices/`: folder containing the aggregate confusion matrices derived from the results of the LOPO CV of each experiment run during benchmarking. Confusion matrices are stored with a path name similar to the predictions: `confusion_matrices/[sec_num]_[validation_data]_Validation_[num_activities]_Activities_[sensor].pdf` (Sec 4.2.1 and 4.2.2) and `confusion_matrices/4.2.3_PSG_Sleep_Labels.pdf`.

`figures/`: folder containing larger copies of the figures in our paper as well as additional figures. For each figure in our paper, a larger PDF is stored at `figures/Fig_[figure_number_from_paper]_[description_of_figure].pdf`.

`paper.pdf`: a preprint of the IMWUT paper this repository accompanies.

### Path Variables
Most of the path variables in this folder follow the same values as in the [`/models/` folder](https://github.com/mHealth-Research-Group/paaws-benchmarking/blob/main/models/README.md), we define the novel variables below.

#### [sec_num_title] and [sec_num]

Defines the sections of the paper the results of this experiment is reported in. Possible values: `4.2.1-Benchmark_SimFL_Lab`, `4.2.2-Benchmark_FL`, and `4.2.3-Benchmark_PSG_Labels`. The variable `[sec_num]` refers to just the section number. Possible values: `4.2.1`, `4.2.2`, and `4.2.3`.

#### [data_used_in_validation]

The data used for validation (this is only relevant in the `4.2.1-Benchmark_SimFL_Lab/` folder where we evaluate models on both the SimFL+Lab data using LOPO CV and trained models on the FL data). Possible values: `SimFL_Lab` and `FL`.

#### [ds_lo]

The ID of the participant's data that generated these predictions. E.g., a file name with `DS_10` has the results of predicting on DS_10's data.

