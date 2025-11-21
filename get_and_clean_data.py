"""
=========================================
Helper script with methods to retrieve the PAAWS accelerometer data from its
location and "clean" the data to only include the specified activities
used throughout our experiments.
=========================================
Author: Veronika K. Potter
Email: potter[dot]v[at]northeastern[dot]edu (potter.v@northeastern.edu)
"""

import config
from datetime import datetime
import numpy as np
import pandas as pd


def get_accel_start_time(file):
    """
    Gets and returns the data collection start time from the accelerometer data
    file.

    Parameters
    ----------
    file : str
        The file location of the accelerometer data.

    Returns
    -------
    start_time : datetime
        The data collection start time.
    """

    odf = pd.read_csv(file, sep=" ", header=None, skiprows=2, nrows=2)
    odf = odf.loc[:, 2]
    st_str = odf[1] + " " + odf[0]
    start_time = datetime.strptime(st_str, "%m/%d/%Y %H:%M:%S")
    return start_time


def get_dataset_accel(ds):
    """
    Gets and returns the accelerometer data and data collection start time
    for a specified participant (e.g., DS_[ID]) and data collection protocol.

    Parameters
    ----------
    ds : int
       The participant's ID (DS_[ID]) to retrieve data from.

    Returns
    -------
    accel : pd.DataFrame
        The participant's accelerometer data for the specified protocol.

    start_time : datetime
        The participant's accelerometer data start time.
    """

    # TODO: Path names should be updated for your environment.
    base_path = "./data/"
    if config["LAB"]:
        # LeftWrist has two sensor locations in lab and FL
        if "LeftWrist" in config["SENSOR"]:
            sensor = "LeftWristTop"
        else:
            sensor = config["SENSOR"]

        acc_path = (
            f"{base_path}PAAWS_SimFL_Lab/DS_{ds}/accel/DS_{ds}-Lab-{sensor}.csv"
        )

    elif config["FL"]:
        acc_path = (
            f"{base_path}"
            f"/PAAWS_FreeLiving/DS_{ds}/accel/"
            f"DS_{ds}-Free-{config['SENSOR']}.csv"
        )

    accel = pd.read_csv(acc_path, skiprows=10)
    start_time = get_accel_start_time(acc_path)

    return accel, start_time


def get_accel():
    """
    Gets and returns the accelerometer data and data collection start time
    for all participants in DATASETS for a specified data collection protocol
    (LAB or FL).

    Parameters
    ----------
    None.

    Returns
    -------
    accel : dict of pd.DataFrame
        Dictionary containing all participants accelerometer data for the
        specified protocol.

        * Keys : int
            Participant IDs.
        * Values : pd.DataFrame
            Dataframes containing the accelerometer data from the specified
            participant and collection protocol.

    accel_starts : dict of datetime
        The participant's accelerometer data start time.

        * Keys : int
            Participant IDs.
        * Values : datetime
            The participant's accelerometer data start time.
    """

    accel = {}
    accel_starts = {}

    for ds in config["DATASETS"]:
        accel[ds], accel_starts[ds] = get_dataset_accel(ds)

    return accel, accel_starts


def map_labels(raw_labels):
    """
    Maps a given set of labels (in the PAAWS annotation taxonomy format) to a
    specified (re-grouped) activity set (utils.MAPPING_SCHEMES).

    Parameters
    ----------
    raw_labels : pd.DataFrame
       A set of labels in the PAAWS annotation taxonomy.

    Returns
    -------
    pd.DataFrame
        A set of labels that is mapped to match and contain only the desired
        activities in the activity grouping scheme (utils.MAPPING_SCHEMES).
    """

    # For each "raw" label, map if it is in the mapping scheme.
    for i in raw_labels.index:
        mapped = False

        # Map by the HLB (if applicable).
        if str(raw_labels.loc[i]["HIGH_LEVEL_BEHAVIOR"]) != "nan":
            hlbs = raw_labels.loc[i]["HIGH_LEVEL_BEHAVIOR"].split("|")
            for hlb in hlbs:
                if hlb in config["ACT_MAPPING"].keys() and not mapped:
                    mapped = True
                    curr_hlb = config["ACT_MAPPING"][hlb]
                    raw_labels.loc[i, "MAPPED_LABEL"] = curr_hlb

        # Map by the PA_Type (if applicable).
        if (
            not mapped
            and raw_labels.loc[i]["PA_TYPE"] in config["ACT_MAPPING"].keys()
        ):
            mapped = True
            pa_key = raw_labels.loc[i]["PA_TYPE"]
            raw_labels.loc[i, "MAPPED_LABEL"] = config["ACT_MAPPING"][pa_key]

    # Remove all labels not specified in the mapping scheme.
    ind = raw_labels[~raw_labels["MAPPED_LABEL"].isin(config["ACT_LIST"])].index
    raw_labels = raw_labels.drop(ind)

    # Return only the time and mapped label.
    return raw_labels[["START_TIME", "STOP_TIME", "MAPPED_LABEL"]]


def get_dataset_labels(ds):
    """
    Gets and returns the labels from a data collection protocol (LAB or FL) for
    a specified participant (ds).

    Parameters
    ----------
    ds : int
       The participant ID (DS_[ID]) to retrieve data from.

    Returns
    -------
    mapped_labels : pd.DataFrame
        The participant's labels for the specified protocol, mapped to match the
        desired activity grouping scheme (utils.MAPPING_SCHEMES).
    """

    # TODO: this path may need to be updated for your environment.
    base_path = "./data/"

    if config["LAB"]:
        label_path = (
            f"{base_path}/PAAWS_SimFL_Lab/DS_{ds}/label/DS_{ds}-Lab-label.csv"
        )
    elif config["FL"]:
        label_path = (
            f"{base_path}/PAAWS_FreeLiving/DS_{ds}/label/DS_{ds}-Free-label.csv"
        )

    labels = pd.read_csv(label_path, parse_dates=["START_TIME", "STOP_TIME"])
    mapped_labels = map_labels(labels)

    return mapped_labels


def resample_labels(og_labels_df, sec_len=1):
    """
    Resample a set of labels (with variable start and stop time) into evenly
    spaced increments of length sec_len.

    Parameters
    ----------
    og_labels_df : pd.DataFrame
        The original, variably spaced labels.

    sec_len : int, optional
        The desired length of the evenly spaced layers.
        Default is 1.

    Returns
    -------
    sec_labels_df : pd.DataFrame
        The resampled labels where each label spans sec_len s.
    """

    og_labels_df["START_TIME"] = og_labels_df["START_TIME"].dt.round("1s")

    temp_times = [
        (pd.date_range(e.START_TIME, e.STOP_TIME, freq="1s"), e.MAPPED_LABEL)
        for e in og_labels_df.itertuples()
    ]
    res = [
        pd.DataFrame({"START_TIME": times, "MAPPED_LABEL": label})
        for times, label in temp_times
    ]

    sec_labels = pd.concat(res, axis=0, ignore_index=True)
    sec_labels.sort_values(by=["START_TIME"], inplace=True)
    sec_labels.set_index("START_TIME", inplace=True)

    sec_labels_df = sec_labels.resample(str(sec_len) + "s").agg(
        {"MAPPED_LABEL": lambda x: x.iat[0] if len(set(x)) == 1 else pd.NA}
    )
    sec_labels_df.reset_index(inplace=True)

    return sec_labels_df


def window_dataset_labels(labels):
    """
    Divide the original, variable length labels into config['T'] s long,
    non-overlapping windows.

    Parameters
    ----------
    labels : pd.DataFrame
        The original, variably spaced labels.

    Returns
    -------
    windowed_labels : pd.DataFrame
        The windowed labels where each window lasts config['T'] s and contains
        data only from the labeled activities we care about.
    """

    labels = resample_labels(labels)

    # Make a window if and only if the window has the same activity throughout
    # the window.
    label_times = []
    for i in range(
        0, len(labels) - (round(config["T"]) - 1), round(config["T"])
    ):
        if all(
            labels["MAPPED_LABEL"][i : i + round(config["T"])]
            == labels["MAPPED_LABEL"][i]
        ):
            label_times.append(labels["START_TIME"][i])

    # Remove all instances when the activity did not remain unchanged
    # throughout the window.
    mapped_labels_ind = labels[labels["START_TIME"].isin(label_times)].index
    windowed_labels = pd.DataFrame(
        {
            "START_TIME": label_times,
            "MAPPED_LABEL": labels["MAPPED_LABEL"][mapped_labels_ind],
        }
    )

    return windowed_labels


def get_labels():
    """
    Gets and returns the labels for all participants in DATASETS for a specified
    data collection protocol (SimFL+Lab or FL). Labels are returned as mapped
    to the desired activity set (utils.MAPPING_SCHEMES) and windowed into
    config['T'] s length windows.

    Parameters
    ----------
    None.

    Returns
    -------
    windowed_labels : dict of pd.DataFrame
        Dictionary containing all participants labels mapped into the desired
        activity groupings for the specified data collection protocol and
        windowed into non-overlapping T s length segments.

        * Keys : int
            Participant IDs.
        * Values : pd.DataFrame
            The participant's labels for the specified protocol, mapped to
            match the desired activity grouping scheme (utils.MAPPING_SCHEMES)
            and windowed into length config['T'] s windows.
    """

    labels = {}

    # Get and map labels.
    for ds in config["DATASETS"]:
        labels[ds] = get_dataset_labels(ds)

    # Window the labels.
    windowed_labels = {}
    for key in labels.keys():
        windowed_labels[key] = window_dataset_labels(labels[key])

    return windowed_labels


def window_dataset_accel(accel, accel_start, window_labs):
    """
    Window the accelerometer data for a given participant into config['T] length
    windows. Removes all accelerometer data that does correspond to a label in
    our desired activity set (see utils.MAPPING_SCHEMES).

    Parameters
    ----------
    accel : pd.DataFrame
        The accelerometer data from the specified participant and collection
        protocol.


    accel_starts : datetime
        The participant's accelerometer data start time.

    window_labs : pd.DataFrame
        The participant's labels for the specified protocol, mapped to match the
        desired activity grouping scheme (utils.MAPPING_SCHEMES) and windowed
        into length config['T'] s windows.

    Returns
    -------
    np.array
        The participant's windowed accelerometer data for the specified protocol
        that consists of only activities we care about.
    """

    # Remove pre-data collection so accel matches label length.
    pre_col = (window_labs.iloc[0]["START_TIME"] - accel_start).total_seconds()
    accel = accel[int(pre_col) * config["FREQ"] :]

    # Truncate accel to be divisible by WINDOW_SIZE increments.
    num_rows_to_keep = (len(accel) // config["WINDOW_SIZE"]) * config[
        "WINDOW_SIZE"
    ]
    accel = accel.head(num_rows_to_keep)

    # Shape accel into WINDOW_SIZE increments.
    accel_array = accel.to_numpy()
    accel_array = np.reshape(
        accel_array, (-1, config["WINDOW_SIZE"], 3)
    )  # 3 = x, y, z.

    # Remove accel data that is not in the mapping scheme.
    indices_we_care_about = []
    for time in window_labs["START_TIME"]:
        ind = int(
            ((time - window_labs.iloc[0]["START_TIME"]).total_seconds())
            / config["T"]
        )
        indices_we_care_about.append(ind)

    indices_we_care_about = np.array(indices_we_care_about)

    # Return only the accel that corresponds to a desired activity label.
    return accel_array[indices_we_care_about]


def window_accel(accel_dict, start_times_dict, windowed_labels_dict):
    """
    Windows the accelerometer data into T length windows. Removes all
    accelerometer data that does correspond to a label in our desired activity
    set (utils.MAPPING_SCHEMES).

    Parameters
    ----------
    accel_dict : dict of pd.DataFrame
        Dictionary containing all participants accelerometer data for the
        specified protocol.

        * Keys : int
            Participant IDs.
        * Values : pd.DataFrame
            Dataframes containing the accelerometer data from the specified
            participant and collection protocol.

    start_times_dict : dict of datetime
        The participant's accelerometer data start time.

        * Keys : int
            Participant IDs.
        * Values : datetime
            The participant's accelerometer data start time.

    windowed_labels_dict : dict of pd.DataFrame
        Dictionary containing all participants labels mapped into the desired
        activity groupings for the specified data collection protocol and
        windowed into non-overlapping T s length segments.

        * Keys : int
            Participant IDs.
        * Values : pd.DataFrame
            The participant's labels for the specified protocol, mapped to
            match the desired activity grouping scheme (utils.MAPPING_SCHEMES)
            and windowed into length T s windows.

    Returns
    -------
    dict of np.array
        Dictionary containing all participants windowed accelerometer data that
        corresponds to the desired activity set.

        * Keys : int
            Participant IDs.
        * Values : np.array
            The participant's windowed accelerometer data for the specified
            protocol that consists of only activities we care about.
    """

    windowed_accel = {}

    for key in accel_dict:
        windowed_accel[key] = window_dataset_accel(
            accel_dict[key], start_times_dict[key], windowed_labels_dict[key]
        )

    # Remove any null values.
    # NOTE: in our experiments, there shouldn't be any. This code is to handle
    # missing values.
    for key in windowed_accel:
        ind = [
            i
            for i in range(len(windowed_accel[key]))
            if np.isnan(windowed_accel[key][i]).any()
        ]
        if len(ind) != 0:
            windowed_accel[key] = np.delete(windowed_accel[key], ind, axis=0)
            windowed_labels_dict[key] = np.delete(
                windowed_labels_dict[key], ind, axis=0
            )

    return windowed_accel, windowed_labels_dict


def get_and_clean_accel_and_labels():
    """
    Gets and windows all the accelerometer and label data. Returns
    non-overlapping windows of accelerometer and label data that contains only
    data of the desired activities (utils.MAPPING_SCHEMES).

    Parameters
    ----------
    None.

    Returns
    -------
    windowed_accel : dict of np.array
        Dictionary containing all participants windowed accelerometer data that
        corresponds to the desired activity set.

        * Keys : int
            Participant IDs.
        * Values : np.array
            The participant's windowed accelerometer data for the specified
            protocol that consists of only activities we care about.

    windowed_labels : dict of pd.DataFrame
        Dictionary containing all participants labels mapped into the desired
        activity groupings for the specified data collection protocol and
        windowed into non-overlapping T s length segments.

        * Keys : int
            Participant IDs.
        * Values : pd.DataFrame
            The participant's labels for the specified protocol, mapped to
            match the desired activity grouping scheme (utils.MAPPING_SCHEMES)
            and windowed into length T s windows.
    """

    accel_dict, accel_starts_dict = get_accel()
    temp_windowed_labels = get_labels()

    windowed_accel, windowed_labels = window_accel(
        accel_dict, accel_starts_dict, temp_windowed_labels
    )

    return windowed_accel, windowed_labels
