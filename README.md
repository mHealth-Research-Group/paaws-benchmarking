# The Physical Activity Assessment Using Wearable Sensors Dataset: Labeled Free-Living Accelerometer Data

This is the GitHub repository of The Physical Activity Assessment Using Wearable Sensors Dataset: Labeled Free-Living Accelerometer Data (IMWUT '25). If you use this data, please [cite our paper](#citation) and [email us](https://www.paawsstudy.org/contact-us.html). All the [data used in these experiments is available](#data-availability).

## Repository Structure

`/data`: folder with two participants' data (raw and as computed features) to use when training or running trained models without downloading the entire dataset. Our code is set up to use paths directing to this folder, if you [download the PAAWS R1 dataset](https://hdl.handle.net/2047/D20806901) and want to use our code, place the full dataset in this directory.

`/models`: folder containing a single random forest model trained on the PAAWS SimFL+Lab data to recognize five activities using the data from 20 participants. To run more of our models, [please download them](https://drive.google.com/drive/folders/12Xr5isM4o_63GQXUstmpLAYKuu1uvIc9?usp=sharing) and unzip them individually in this folder.

`/results`: folder containing (1) the predictions from all the experiments in our paper, (2) all the aggregate confusion matrices from each experiment, (3) larger copies of the paper figures, and (4) a preprint of our paper.

`/replicate_our_results`: folder containing information on [replicating the results from our benchmarking experiments](#replicating-our-results) (Sec 4.3.1-4.3.3).

`compute_features.py`: helper python file to run compute features while preprocessing accelerometer data.

`get_and_clean_data.py`: helper python file to (1) get raw data, (2) segment/window the data, and (3) transform the labels into the ones specified via the `--mapping` flag.

`make_predictions.py`: python script to make predictions using a pre-trained model from our paper. To run this script, you must have unzipped the subfolder in `/models`. (This script takes the place of `main.py`.)

`run_experiment.py`: python script to run an experiment from our paper (i.e., script to train and evaluate a random forest using the PAAWS data). (This script takes the place of `main.py`.)

`train_eval_model.py`: helper python file to train and (if applicable) evaluate a random forest model using the PAAWS data.

`utils.py`: helper python file containing dictionaries used in `run_experiment.py` and `make_predictions.py`.

**NOTE**: Running any of the code in this repo (especially on more participants data) can require a lot of compute resources. We would recommend running these jobs on a high performance cluster with adequate compute power.

#### A Note on `main.py`

Our code is currently set up to not use a `main.py` for increased readability. The scripts in this repo that take the place of `main.py` are `run_experiment.py` and `make_predictions.py`.

### Training and Evaluating a Model
Models are trained and evaluated using the `/run_experiment.py` script.
1. Clone the repository.
```
git clone git@github.com:mHealth-Research-Group/paaws-benchmarking.git
```
2. (Optional) Set up a virtual environment of your choice. For our experiments, we used **Python version 3.13.5** (see `pyproject.toml` and `requirements.txt` for more information on our specific environment).
3. Install the required python packages in `requirements.txt`. We recommend using pip.
```
pip install -r requirements.txt
```
4.  Train a new random forest (RF) model using the sample PAAWS data provided in this repo. In this example, we are using the Left Wrist, SimFL+Lab data from DS_36 as training data and evaluate the trained RF on DS_10's Left Wrist SimFL+Lab data. If you'd like to train or evaluate a model using more data than the small subset we provide in this repo, update the `/data` folder to contain the entirety of the dataset or alter the pathing in the preprocessing code (paths to where the data is fetched from are on lines 60 and 183 in `get_and_clean_data.py`). NOTE: when changing what data is used, you should update the command line arguments accordingly (see the **NOTE**).
```
python run_experiment.py --ds_lo=10 --sensor="LeftWristTop" --dataset="2" --lab --out_file="temp_results"
```

**NOTE**: We use command line arguments to toggle settings in each experiment. For more information on what each command line argument means and how to configure your arguments, run the following line:
```
python run_experiment.py --h
```

### Using our Models for Inference

To make predictions using a pre-trained model run the `/make_predictions.py` script. In this example, we are using the random forest trained on 20 participants Left Wrist data collected during the SimFL+Lab protocol and evaluating on the data collected from DS_10 (**NOTE**: this dataset is included in the training set and should not be used to evaluate the performance of this model; we chose this dataset for demonstration purposes). We've noted in the code where you will need to change the processing if you'd like to evaluate a model using different data (`/make_predictions.py` line 151). Make sure you have [unzipped the model weights](https://github.com/mHealth-Research-Group/paaws-benchmarking/blob/main/models/README.md#using-the-example-model) before running this code.
```
python make_predictions.py --ds_lo=10 --sensor="LeftWrist" --n="20" --mapping="lab_fl_5" --out_file="temp_results" --lab
```

**NOTE**: For making inferences, we also use command line arguments to toggle settings in each experiment. For more information on what each command line argument means and how to configure your arguments, run the following line:
```
python make_predictions.py --h
```

### Replicating Our Results
We've included `/replicate_our_results/replicate_4.2.1_simfl_jobs.txt` and `/replicate_our_results/replicate_4.2.2_fl_jobs.txt` which contains the bash commands we ran when reporting the results in our paper. We recommend running each of these commands as separate, independent jobs on a cluster space with a large amount of compute space. [This document](https://github.com/mHealth-Research-Group/paaws-benchmarking/blob/main/replicate_our_results/replicate_our_results.md) contains more information on replicating our results.

## Data Availability
Our data is available for download ([download the PAAWS R1 dataset](https://hdl.handle.net/2047/D20806901)). We have written accompanying codebooks for [data users](https://docs.google.com/document/d/1NBHiTc89rqZIpqk-gRAcRLGijC48WoBa/edit?usp=sharing&ouid=108613616105994133659&rtpof=true&sd=true) and about our [data collection protocols](https://docs.google.com/document/d/1kgi7MNqh516IOvbND5aj7rMhJ-_FHNIzrDQDF_l2Spc/edit?usp=sharing).

## Citation
If you use our code or dataset please cite:

Veronika Potter, Hoan Tran, Daniel Mobley, Suzanne M. Bertisch, Dinesh John, and Stephen Intille. 2025. The Physical Activity
Assessment Using Wearable Sensors (PAAWS) Dataset: Labeled Laboratory and Free-Living Accelerometer Data. *Proc. ACM
Interact. Mob. Wearable Ubiquitous Technol.* 9, 4, Article 204 (December 2025), 32 pages. [https://doi.org/10.1145/3770639](https://doi.org/10.1145/3770639)

We also provide our citation as a bibtex:
```bibtex
@article{potter_2025__paaws_dataset,
  title = {The Physical Activity Assessment Using Wearable Sensors (PAAWS) Dataset: Labeled Laboratory and Free-Living Accelerometer Data},
  author = {Potter, Veronika, and Tran, Hoan and Mobley, Daniel and Bertisch, Suzanne M. and John, Dinesh and Intille, Stephen},
  year = {2025},
  month = Dec,
  journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
  volume = {9},
  number = {4},
  doi = {10.1145/3770639}
}
```

Our paper is available as a [pdf](https://github.com/mHealth-Research-Group/paaws-benchmarking/blob/main/results/paper.pdf) or from the [ACM Digital library](https://doi.org/10.1145/3770639).

## More Info on PAAWS

For more information on the PAAWS dataset and resources associated with the dataset, see [the PAAWS study repository](https://github.com/mHealth-Research-Group/paaws-study).

## Questions, Comments, Issues

Please use the [issue tracker](https://github.com/mHealth-Research-Group/paaws-study/issuess) for any questions, comments, or issues when using the data.
If we have not responded to your issue within a week, please [email us](https://www.paawsstudy.org/contact-us.html).

## Acknowledgements

Research reported in this publication was supported, in part, by the National Cancer Institute of the National Institutes of Health under award number R01CA252966. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.