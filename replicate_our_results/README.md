# Replicate Our Results

To replicate our results, we provide a code snippet that generates a .txt file containing all the commands (e.g., bash commands and arguments) that we ran while benchmarking the PAAWS R1 dataset. *To replicate our results, we recommend running each of the commands as an individual job on a high performance cluster (HPC).* Many of the commands use a large amount of RAM.

**NOTE**: To reduce the amount of RAM used and expedite the training process of these random forests, we recommend preprocessing all the data into features and mapped labels prior to training any models. Then, when training a model, it is easy to alter the `run_experiment.py` script to read in an existing, saved preprocessed dictionary or NumPy arrays instead of preprocessing the raw data with every job.

## 4.2.1 - Benchmarking the SimFL+Lab Dataset

Run all the commands to replicate the results in Sec 4.2.1 by altering the following snippet of python code (used to generate the txt). We recommend altering the code to work for your specific HPC environment. Additionally, we've attached a .txt file `/replicate_4.2.1_simfl_jobs.txt` containing all the commands run to benchmark the SimFL+Lab dataset.

```python
from utils import DATASET_LISTS

sensors = ["LeftWristTop", "RightThigh"]
mapping_lists = ["lab_fl_5", "lab_fl_9", "lab_42"]
dataset_lists = ["SimFL_20", "126_1", "126_2","126_3", "126_4","126_5", "252"]

st_to_write = "***** Replicate Sec 4.2.1 Benchmarking the PAAWS SimFL+Lab Dataset *****\n\n"

for sensor in sensors:
    for mapping in mapping_lists:
        for dataset in dataset_lists:
            st_to_write += f"===== Commands for LOPO CV for {sensor}, {mapping} Activities, {dataset} Participants \n"

            for participant in DATASET_LISTS[dataset]:
                st_to_write += f"python3 run_experiment.py --ds_lo={participant} --sensor={sensor} --dataset={dataset} --mapping={mapping} --out_file=\'replicated_results\' --lab \n"

            st_to_write += "\n"

with open("replicate_4.2.1_simfl_jobs.txt", "w") as f:
  print(f.write(st_to_write))

```

## 4.2.2 - Benchmarking the FL Dataset
Run all the commands to replicate the results in Sec 4.2.2 by altering the following snippet of python code (used to generate the txt). We recommend altering the code to work for your specific HPC environment. Additionally, we've attached a .txt file `/replicate_4.2.2_fl_jobs.txt` containing all the commands run to benchmark the SimFL+Lab dataset.

```python
from utils import DATASET_LISTS

sensors = ["LeftWrist", "RightThigh"]
mapping_lists = ["lab_fl_5", "fl_11", ]
dataset_lists = ["FL_20"]

st_to_write = "***** Replicate Sec 4.2.2 Benchmarking the PAAWS R1 FL Dataset *****\n\n"

for sensor in sensors:
    for mapping in mapping_lists:
        for dataset in dataset_lists:
            st_to_write += f"===== Commands for LOPO CV for {sensor}, {mapping} Activities, {dataset} Participants \n"

            for participant in DATASET_LISTS[dataset]:
                st_to_write += f"python3 run_experiment.py --ds_lo={participant} --sensor={sensor} --dataset={dataset} --mapping={mapping} --out_file=\'replicated_results\' \n"

            st_to_write += "\n"

with open("replicate_4.2.2_fl_jobs.txt", "w") as f:
  print(f.write(st_to_write))
```

## 4.2.3 - Benchmarking the PSG Dataset

Run the <a href="https://github.com/binodthapachhetry/SWaN" target="_blank">SWaN algorithm</a> across the FL participants data (`DATASET_LISTS['FL_20']` in `/utils.py`).

**NOTE**: Some of the participants in the PAAWS R1 FL dataset do not have sleep data, we ran our analyses across only the participants that had sleep data.
