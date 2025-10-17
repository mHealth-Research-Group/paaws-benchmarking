"""
=========================================
Run inference on a dataset using one of our pre-trained models.
=========================================
Author: Veronika K. Potter
Email: potter[dot]v[at]northeastern[dot]edu (potter.v@northeastern.edu)
"""

import argparse
import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from utils import MAPPING_SCHEMES


def parse_arguements():
    """
    Parses all the arguments from the command line. Specifies help for each
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
        default="10",
        help=(
            "The participant's data (i.e., DS_ID) to predict on. Defaults to "
            "10 for DS_10."
        ),
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="LeftWrist",
        help=(
            "The sensor to use (e.g., LeftWrist, RightThigh). Defaults to "
            "LeftWrist"
        ),
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="temp_results",
        help=(
            "The output file path to save results to. Defaults to temp_results/."
        ),
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="lab_fl_5",
        help=(
            "The activity mapping scheme to use (e.g., keys of "
            "utils.MAPPING_SCHEMES). Defaults to "
            "utils.MAPPING_SCHEMES['lab_fl_5']."
        ),
    )
    parser.add_argument(
        "--n",
        type=str,
        default="FL_20",
        help=(
            "The number of participants' data a model was trained using. "
            "Defaults to FL_20 (i.e., utils.DATASET_LISTS['FL_20']). See "
            "DATASET_LISTS in utils.py for more details."
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
        The output file path to save results to (as csv or joblib files).

    Returns
    -------
    None.
    """

    config_dict = {
        # Data to be predicted on variables.
        "DS_LO": args.ds_lo,
        "DATASETS": [args.ds_lo],
        "SENSOR": args.sensor,
        "LAB": bool(args.lab),
        "FL": not bool(args.lab),
        "ACT_MAPPING": MAPPING_SCHEMES[args.mapping],
        "FREQ": args.f,
        "T": args.t,
        "WINDOW_SIZE": args.f * args.t,

        # Additional model variables.
        "NUM_PARTICIPANTS": args.n,
        "NUM_ACTS": len(set(list(MAPPING_SCHEMES[args.mapping].values()))),
        "ACT_LIST": set(list(MAPPING_SCHEMES[args.mapping].values())),

        # Output file path.
        "OUT_FILE": out_file,
    }

    # Additional model variables. Models trained on SimFL+Lab data have the
    # prefix "SimFL" in their file names.
    if bool(args.lab):
        config_dict["PROTOCOL"] = "SimFL_Lab"
    else:
        config_dict["PROTOCOL"] = "FL"

    return config_dict


def make_predictions():
    """
    Retrieves data to make new inferences on. Segments and parses the data.
    Loads pre-saved model and scalar weights. Makes new inferences using the
    specified model.

    Parameters
    ----------
    None.

    Returns
    -------
    None. All results saves to csv files.
    """

    # Import inside function to use config as a global variable.
    from get_and_clean_data import get_and_clean_accel_and_labels
    from compute_features import make_features

    # Load pretrained model weights and scaler to reshape the data
    # TODO: you may need to change these paths to your data.
    file = (
        f"./models/{config['PROTOCOL']}/{config['NUM_ACTS']}_Activities/"
        f"{config['SENSOR']}_{config['NUM_PARTICIPANTS']}_Participants/"
        f"{config['PROTOCOL']}_{config['SENSOR']}_"
        f"{config['NUM_ACTS']}_Acts"
        f"_{config['NUM_PARTICIPANTS']}_Participants"
    )

    print(("\n***** Step 1 of 3: "
           "Loading weights for the RF and scaler. *****"))
    rf = joblib.load(f"{file}_RF.joblib")
    scaler = joblib.load(f"{file}_SCALER.joblib")

    # Get and compute features for the data to make new predictions from.
    # NOTE: This script predicts on *cleaned* data. I.e., in our experiments,
    # we predicted on data that was *only* comprised of activities we were
    # trying to classify. You may need to modify the code accordingly to add in
    # additional activities beyond those in config["ACT_MAPPING"].
    print(("\n***** Step 2 of 3: "
           "Getting evaluation data and computing features. *****"))
    ds_accel, ds_labels = get_and_clean_accel_and_labels()
    ds_features = make_features(ds_accel[config["DS_LO"]])

    print((f"\n* Evaluation set size: "
           f"{ds_features.shape[0]} samples"
           f" x {ds_features.shape[1]} features."))

    # Make and save predictions.
    print(("\n***** Step 3 of 3: "
           "Making predictions on the evaluation set. *****"))
    scaled_accel = scaler.transform(ds_features.to_numpy())
    predictions = rf.predict(scaled_accel)

    true_pred_df = pd.DataFrame(
        data=ds_labels[config["DS_LO"]], columns=["START_TIME", "TRUE"]
    )
    true_pred_df["TRUE"] = ds_labels[config["DS_LO"]]["MAPPED_LABEL"]
    true_pred_df["PREDS"] = predictions
    save_path = (f"{config['OUT_FILE']}/{config['PROTOCOL']}_"
                 f"{config['SENSOR']}_{config['NUM_ACTS']}_Acts_"
                 f"{config['NUM_PARTICIPANTS']}_Participants_DS_"
                 f"{config['DATASETS'][0]}_PREDS.csv")

    print(f"\n* Saving results to {save_path}.")
    true_pred_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    command_args = parse_arguements()

    # Set up output pathing.
    # TODO: For your specific environment, you may want to change this.
    path = os.getcwd()
    OUT_FILE = f"{path}/{command_args.out_file}"

    if not os.path.exists(OUT_FILE):
        os.makedirs(OUT_FILE)

    config = set_up_config(command_args, OUT_FILE)
    sys.modules["config"] = config

    make_predictions()

    # Output config and completion status.
    print("\n\n***** Job finished successfully! *****\n")
    print("== Config ==")
    for name, value in config.items():
        if not name.startswith("_"):
            print(f"{name}: {value}")
