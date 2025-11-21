"""
=========================================
Helper script with methods to make training sets, train, evaluate, and/or save
the random forest (RF) algorithm we use in our experiments.
=========================================
Author: Veronika K. Potter
Email: potter[dot]v[at]northeastern[dot]edu (potter.v@northeastern.edu)
"""

import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import pandas as pd


def make_training_sets_from_np(acc_features_dict, windowed_labels):
    """
    From the feature sets of the accel data and their respective labels, make
    training sets of accelerometer data and labels to train/evaluate a model.

    Parameters
    ----------
    acc_features_dict : dict of np.array or pd.DataFrame
        The features computed from the windowed accelerometer data by
        participant.

        * Keys : int
            Participant IDs.
        * Values : np.array or pd.DataFrame
            The participant's windowed accelerometer data computed into
            features for the specified protocol that consists of only
            activities we care about.


    windowed_labels : dict of np.array or pd.DataFrame
        The labels from a single participant.

        * Keys : int
            Participant IDs.
        * Values : np.array or pd.DataFrame
            The participant's labels for the activities we care about.

    Returns
    -------
    training_accel : np.array
        A single array containing all the features computed from all
        participants in the training sets accelerometer data to be used for
        training the RF.

    training_labels : np.array
        A single array containing all the labels from all participants in the
        training sets accelerometer data to be used for training the RF.
    """

    training_accel = None
    training_labels = None

    # Concat all "MAPPED_LABEL"s and all accel in the same order.
    for key in acc_features_dict:
        # Remove the "left out" dataset for LOPO CV.
        if int(key) != int(config["DS_LO"]):
            if training_accel is None:
                training_accel = acc_features_dict[key].to_numpy()
                if isinstance(windowed_labels[key], pd.DataFrame):
                    curr_labs = windowed_labels[key]["MAPPED_LABEL"].to_numpy()
                    training_labels = curr_labs
                elif isinstance(windowed_labels[key], np.array):
                    training_labels = windowed_labels[key][:, 1]
            else:
                curr_acc = acc_features_dict[key].to_numpy()
                training_accel = np.vstack([training_accel, curr_acc])
                if isinstance(windowed_labels[key], pd.DataFrame):
                    to_append = windowed_labels[key]["MAPPED_LABEL"].to_numpy()
                elif isinstance(windowed_labels[key], np.array):
                    to_append = windowed_labels[key][:, 1]

                training_labels = np.concatenate([training_labels, to_append])

    # Make a training set comprised of ALL data (i.e., no DS is left-out).
    if config["ALL_DATA"]:
        training_accel = np.vstack(
            [training_accel, acc_features_dict[config["DS_LO"]].to_numpy()]
        )

        if isinstance(windowed_labels[config["DS_LO"]], pd.DataFrame):
            to_append = windowed_labels[config["DS_LO"]][
                "MAPPED_LABEL"
            ].to_numpy()
        elif isinstance(windowed_labels[config["DS_LO"]], np.array):
            to_append = windowed_labels[config["DS_LO"]][:, 1]

        training_labels = np.concatenate([training_labels, to_append])

    print((f"\n* Training set size: {training_accel.shape[0]} samples"
          f" x {training_accel.shape[1]} features."))
    if not config["ALL_DATA"]:
        print((f"* Test set size: "
               f"{acc_features_dict[config['DS_LO']].shape[0]} samples"
               f" x {acc_features_dict[config['DS_LO']].shape[1]} features."))

    return training_accel, training_labels


def train_eval_RF(training_accel, training_labels, eval_accel, eval_labels):
    """
    Trains and evaluates the RF used in our experiments. Optionally saves the
    trained model's weights.

    Parameters
    ----------
    training_accel : np.array
        The accelerometer data to be used for training the RF.

    training_labels : np.array
        The labels to be used when training the RF.

    eval_accel : np.array
        The accelerometer data to be used for evaluating the RF. In our
        experiments, this is the DS_LO's accelerometer data.

    eval_labels : np.array
        The labels to be used for evaluating the RF. In our
        experiments, this is the DS_LO's labels.

    Returns
    -------
    None. All predictions or saved weights are exported as files.
    """

    # Normalize features.
    # NOTE: To prevent overfitting to the test set, we use only the training
    # data to define the scaler.

    scaler = StandardScaler()
    scaler.fit(training_accel)

    train_accel_norm = scaler.transform(training_accel)
    eval_accel_norm = scaler.transform(eval_accel.to_numpy())

    # Train the RF and predict the evaluation data.
    rf = RandomForestClassifier(n_estimators=1000, verbose=1)
    rf = rf.fit(train_accel_norm, training_labels)
    rf_preds = rf.predict(eval_accel_norm)

    print("\n* Training complete.")

    # If all data is in the training set, save model. Else, save the DS_LO
    # predictions with ground truth labels for LOPO CV.
    if config["LAB"]:
        protocol = "SimFL_Lab"
    else:
        protocol = "FL"

    if config["SAVE_MODEL"]:
        save_path = (f"{config["OUT_FILE"]}{protocol}_{config['SENSOR']}_"
                     f"{config['NUM_ACTS']}_Acts_{config['DATASET']}_"
                     f"Participants")

        print(f"\nSaving model to {save_path}_[RF/SCALER].joblib.")

        joblib.dump(rf, f"{save_path}_RF.joblib")
        joblib.dump(scaler, f"{save_path}_SCALER.joblib")
    else:
        res_df = pd.DataFrame(
            data=eval_labels, columns=["START_TIME", "MAPPED_LABEL"]
        )
        res_df.loc[:, "PREDICTION"] = rf_preds
        save_path = (f"{config['OUT_FILE']}{protocol}_{config['SENSOR']}_"
                     f"{config['NUM_ACTS']}_Acts_{config['DATASET']}_"
                     f"Participants_DS_{config['DS_LO']}.csv")

        print("\n* Evaluating the model.")
        print(f"* Saving results to {save_path}.")

        res_df.to_csv(f"{save_path}")
