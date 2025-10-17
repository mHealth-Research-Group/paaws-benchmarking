"""
=========================================
Run one of the benchmarking experiments using the PAAWS SimFL+Lab or FL
accelerometer data.
=========================================
Author: Veronika K. Potter
Email: potter[dot]v[at]northeastern[dot]edu (potter.v@northeastern.edu)
"""

import argparse
import os
import sys
from tqdm import tqdm
from utils import MAPPING_SCHEMES, DATASET_LISTS


def parse_arguements():
    """
    Parses all the arguments for the command line. Specifies help for each
    flag.

    Parameters
    ----------
    None.

    Returns
    -------
    args : argparse.Namespace
        The arguments parsed from the command line.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ds_lo",
        type=int,
        default=10,
        help=(
            "The dataset (e.g., the int for DS_[ID]) to be left-out in"
            "leave-one-participant-out cross validation. Defaults to 10 for"
            "DS_10."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="FL_20",
        help=(
            "The participant set to use (keys of utils.DATASET_LIST). Defaults "
            "to utils.DATASET_LIST['FL_20']"
        ),
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="LeftWrist",
        help=(
            "The sensor to use (e.g., LeftWrist, RightThigh). Defaults to"
            " LeftWrist."
        ),
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="lab_fl_5",
        help=(
            "The activity mapping scheme to use (e.g., keys of "
            "utils.MAPPING_SCHEMES). Defaults to"
            "utils.MAPPING_SCHEMES['lab_fl_5']."
        ),
    )
    parser.add_argument(
        "--t",
        type=float,
        default=10,
        help="The window length in seconds. Defaults to 10 s windows.",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=80,
        help="The frequency of the data in Hz. Defaults to 80 Hz.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="temp_results",
        help=(
            "The output file path to save results to. Defaults to "
            "temp_results/."
        ),
    )
    parser.add_argument(
        "--lab",
        action="store_true",
        default=False,
        help=(
            "If the data to be used is from the SimFL+Lab data or not. "
            "If flagged use lab data. The data used defaults to the FL data."
        ),
    )
    parser.add_argument(
        "--all_data",
        action="store_true",
        default=False,
        help=(
            "If the data to be used in training should *not* leave a "
            "participant out. If flagged *do not* leave out [ds_lo]'s data "
            "from the training set. Defaults to False (i.e., leaving [ds_lo]'s"
            " data out of the training set)."
        ),
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help=(
            "If the weights of the model should be saved. If flagged "
            "save the model. Defaults to False (i.e., not saving the model)."
        ),
    )

    parsed_args = parser.parse_args()

    return parsed_args


def set_up_config(args, out_file):
    """
    Sets up the config.py file using the specified arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments parsed from the command line.

    out_file : str
        The output file path to save results to (as CSV or joblib files).

    Returns
    -------
    None.
    """

    config_dict = {
        # Training/evaluation set variables.
        "DS_LO": args.ds_lo,
        "DATASET": args.dataset,
        "DATASETS": DATASET_LISTS[args.dataset],

        # Data property variables.
        "FREQ": args.f,
        "T": args.t,
        "WINDOW_SIZE": args.f * args.t,
        "SENSOR": args.sensor,

        # Data collection protocol specification.
        "LAB": bool(args.lab),
        "FL": not bool(args.lab),

        # Label set/activity grouping specifications.
        "ACT_MAPPING": MAPPING_SCHEMES[args.mapping],
        "ACT_LIST": set(list(MAPPING_SCHEMES[args.mapping].values())),
        "NUM_ACTS": len(set(list(MAPPING_SCHEMES[args.mapping].values()))),

        # Output file path.
        "OUT_FILE": out_file,
        "SAVE_MODEL": args.save_model,
        "ALL_DATA": args.all_data,
    }

    return config_dict


def run_experiment():
    """
    Retrieves all accelerometer and label data for the specified participants
    and data collection protocol. Preprocesses the data into windows and
    computes feature values. Trains and evaluates a random forest algorithm
    with the data using leave-one-out cross-validation if save_model == False.

    Parameters
    ----------
    None.

    Returns
    -------
    None. All results saved to csv files and model weight (if applicable) saved
    to a joblib file.
    """

    # Import inside function to use config as a global variable.
    from get_and_clean_data import (
        get_dataset_accel,
        get_dataset_labels,
        window_dataset_accel,
        window_dataset_labels
    )
    from compute_features import make_features
    from train_eval_model import train_eval_RF, make_training_sets_from_np

    features = {}
    windowed_labels = {}

    print(("\n***** Step 1 of 3: "
           "Getting neccesary data from database and computing features. *****"))

    for ds in tqdm(config["DATASETS"], unit="participant"):
        accel, accel_start = get_dataset_accel(ds)
        labels = get_dataset_labels(ds)

        windowed_label = window_dataset_labels(labels)
        windowed_accel = window_dataset_accel(
            accel,
            accel_start,
            windowed_label)

        features[ds] = make_features(windowed_accel)
        windowed_labels[ds] = windowed_label

        del accel, accel_start, labels, windowed_accel

    print(("\n***** Step 2 of 3: "
           "Create training set and (if applicable) evaluation set. *****"))

    training_accel, training_labels = make_training_sets_from_np(
        features, windowed_labels
    )

    print(("\n***** Step 3 of 3: "
           "Training the RF and (if applicable) evaluating. "
           "This may take several minutes. *****\n"))
    train_eval_RF(
        training_accel,
        training_labels,
        features[config["DS_LO"]],
        windowed_labels[config["DS_LO"]],
    )


if __name__ == "__main__":
    print("***** Job started! *****")
    command_args = parse_arguements()

    # Set up output pathing.
    # TODO: For your specific environment, you may want to change this.
    path = os.getcwd()
    OUT_FILE = f"{path}/{command_args.out_file}/"

    if not os.path.exists(OUT_FILE):
        os.makedirs(OUT_FILE)

    config = set_up_config(command_args, OUT_FILE)
    sys.modules["config"] = config

    run_experiment()

    # Output config and completion status.
    print("\n\n***** Job finished successfully! *****\n")
    print("== Config ==")
    for name, value in config.items():
        if not name.startswith("_"):
            print(f"{name}: {value}")
