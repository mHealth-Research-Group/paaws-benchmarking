"""
=========================================
Helper script with methods to compute individual features and the final feature
set used in our experiments.
=========================================
Author: Veronika K. Potter
Email: potter[dot]v[at]northeastern[dot]edu (potter.v@northeastern.edu)
"""

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy import stats
import config
from scipy.signal import butter, sosfiltfilt


def compute_min(curr_features, no_noise_acc, axes_map={"x": 0, "y": 1, "z": 2}):
    """
    From the given windowed single-axis, accelerometer data, without DC
    component or high frequency noise, compute the min value across the
    specified axes.

    Parameters
    ----------
    curr_features : pd.DataFrame
        The current features that have been already computed.

    no_noise_acc : np.array
        The accelerometer data from a single participant with noise removed.

    axes_map : dict of int, optional
        Denotes if the passed in no_noise_acc data is the vector magnitude or
        the triaxial data.

        * Keys : string
            The axis or "vm".
        * Value : int
            The axis of the desired data to compute a feature from.

    Returns
    -------
    curr_features : pd.DataFrame
        The features that have been already computed plus the computed min.
    """

    if len(axes_map) == 1:
        temp_df = pd.DataFrame(no_noise_acc.T)
        curr_features["ACC_min_vm"] = temp_df.apply(lambda x: x.min())
    else:
        for axis in axes_map:
            temp_df = pd.DataFrame(no_noise_acc[:, :, axes_map[axis]].T)
            curr_features[f"ACC_min_{axis}"] = temp_df.apply(lambda x: x.min())

    return curr_features


def compute_max(curr_features, no_noise_acc, axes_map={"x": 0, "y": 1, "z": 2}):
    """
    From the given windowed accelerometer data, without DC
    component or high frequency noise, compute the variance max by axis.

    Parameters
    ----------
    curr_features : pd.DataFrame
        The current features that have been already computed.

    no_noise_acc : np.array
        The accelerometer data from a single participant with noise removed.

    axes_map : Dict of int, optional
        Denotes if the passed in no_noise_acc data is the vector magnitude.
        Defaults to using tri-axial accelerometer data.

        * Keys : string
            The given axis.

        * Values : int
            The given axis as an integer.

    Returns
    -------
    curr_features : pd.DataFrame
        The features that have been already computed plus the additional
        max value.
    """

    if len(axes_map) == 1:
        temp_df = pd.DataFrame(no_noise_acc.T)
        curr_features["ACC_max_vm"] = temp_df.apply(lambda x: x.max())
    else:
        for axis in axes_map:
            temp_df = pd.DataFrame(no_noise_acc[:, :, axes_map[axis]].T)
            curr_features[f"ACC_max_{axis}"] = temp_df.apply(lambda x: x.max())
    return curr_features


def compute_mean(
    curr_features, no_noise_acc, axes_map={"x": 0, "y": 1, "z": 2}
):
    """
    From the given windowed accelerometer data, without DC
    component or high frequency noise, compute the mean value by axis.

    Parameters
    ----------
    curr_features : pd.DataFrame
        The current features that have been already computed.

    no_noise_acc : np.array
        The accelerometer data from a single participant with noise removed.

    axes_map : Dict of int, optional
        Denotes if the passed in no_noise_acc data is the vector magnitude.
        Defaults to using tri-axial accelerometer data.

        * Keys : string
            The given axis.

        * Values : int
            The given axis as an integer.

    Returns
    -------
    curr_features : pd.DataFrame
        The features that have been already computed plus the additional
        mean value features.
    """

    if len(axes_map) == 1:
        temp_df = pd.DataFrame(no_noise_acc.T)
        curr_features["ACC_mean_vm"] = temp_df.apply(lambda x: x.mean())
    else:
        for axis in axes_map:
            temp_df = pd.DataFrame(no_noise_acc[:, :, axes_map[axis]].T)
            mean_df = temp_df.apply(lambda x: x.mean())
            curr_features[f"ACC_mean_{axis}"] = mean_df
    return curr_features


def compute_var(curr_features, no_noise_acc, axes_map={"x": 0, "y": 1, "z": 2}):
    """
    From the given windowed accelerometer data, without DC
    component or high frequency noise, compute the variance value by axis.

    Parameters
    ----------
    curr_features : pd.DataFrame
        The current features that have been already computed.

    no_noise_acc : np.array
        The accelerometer data from a single participant with noise removed.

    axes_map : Dict of int, optional
        Denotes if the passed in no_noise_acc data is the vector magnitude.
        Defaults to using tri-axial accelerometer data.

        * Keys : string
            The given axis.

        * Values : int
            The given axis as an integer.

    Returns
    -------
    curr_features : pd.DataFrame
        The features that have been already computed plus the additional
        variance features.
    """

    if len(axes_map) == 1:
        temp_df = pd.DataFrame(no_noise_acc.T)
        curr_features["ACC_var_vm"] = temp_df.apply(lambda x: x.var())
    else:
        for axis in axes_map:
            temp_df = pd.DataFrame(no_noise_acc[:, :, axes_map[axis]].T)
            curr_features[f"ACC_var_{axis}"] = temp_df.apply(lambda x: x.var())
    return curr_features


def compute_std(curr_features, no_noise_acc, axes_map={"x": 0, "y": 1, "z": 2}):
    """
    From the given windowed accelerometer data, without DC
    component or high frequency noise, compute the standard deviation value by
    axis.

    Parameters
    ----------
    curr_features : pd.DataFrame
        The current features that have been already computed.

    no_noise_acc : np.array
        The accelerometer data from a single participant with noise removed.

    axes_map : Dict of int, optional
        Denotes if the passed in no_noise_acc data is the vector magnitude.
        Defaults to using tri-axial accelerometer data.

        * Keys : string
            The given axis.

        * Values : int
            The given axis as an integer.

    Returns
    -------
    curr_features : pd.DataFrame
        The features that have been already computed plus the additional
        standard deviation features.
    """

    if len(axes_map) == 1:
        temp_df = pd.DataFrame(no_noise_acc.T)
        curr_features["ACC_std_vm"] = temp_df.apply(lambda x: x.std())
    else:
        for axis in axes_map:
            temp_df = pd.DataFrame(no_noise_acc[:, :, axes_map[axis]].T)
            curr_features[f"ACC_std_{axis}"] = temp_df.apply(lambda x: x.std())
    return curr_features


def compute_skew(
    curr_features,
    no_noise_acc,
    axes_map={"x": 0, "y": 1, "z": 2}
):
    """
    From the given windowed accelerometer data, without DC
    component or high frequency noise, compute the skewness value by axis.

    Parameters
    ----------
    curr_features : pd.DataFrame
        The current features that have been already computed.

    no_noise_acc : np.array
        The accelerometer data from a single participant with noise removed.

    axes_map : Dict of int, optional
        Denotes if the passed in no_noise_acc data is the vector magnitude.
        Defaults to using tri-axial accelerometer data.

        * Keys : string
            The given axis.

        * Values : int
            The given axis as an integer.

    Returns
    -------
    curr_features : pd.DataFrame
        The features that have been already computed plus the additional
        skewness features.
    """

    if len(axes_map) == 1:
        temp_df = pd.DataFrame(no_noise_acc.T)
        curr_features["ACC_skew_vm"] = temp_df.apply(lambda x: stats.skew(x))
    else:
        for axis in axes_map:
            temp_df = pd.DataFrame(no_noise_acc[:, :, axes_map[axis]].T)
            skew_df = temp_df.apply(lambda x: stats.skew(x))
            curr_features[f"ACC_skew_{axis}"] = skew_df
    return curr_features


def compute_kurt(
    curr_features,
    no_noise_acc,
    axes_map={"x": 0, "y": 1, "z": 2}
):
    """
    From the given windowed accelerometer data, without DC
    component or high frequency noise, compute the kurtosis value by axis.

    Parameters
    ----------
    curr_features : pd.DataFrame
        The current features that have been already computed.

    no_noise_acc : np.array
        The accelerometer data from a single participant with noise removed.

    axes_map : Dict of int, optional
        Denotes if the passed in no_noise_acc data is the vector magnitude.
        Defaults to using tri-axial accelerometer data.

        * Keys : string
            The given axis.

        * Values : int
            The given axis as an integer.

    Returns
    -------
    curr_features : pd.DataFrame
        The features that have been already computed plus the additional
        kurtosis features.
    """

    if len(axes_map) == 1:
        temp_df = pd.DataFrame(no_noise_acc.T)
        kurt_df = temp_df.apply(lambda x: stats.kurtosis(x))
        curr_features["ACC_kurt_vm"] = kurt_df
    else:
        for axis in axes_map:
            temp_df = pd.DataFrame(no_noise_acc[:, :, axes_map[axis]].T)
            kurt_df = temp_df.apply(lambda x: stats.kurtosis(x))
            curr_features[f"ACC_kurt_{axis}"] = kurt_df
    return curr_features


def compute_25(curr_features, no_noise_acc, axes_map={"x": 0, "y": 1, "z": 2}):
    """
    From the given windowed accelerometer data, without DC
    component or high frequency noise, compute the 25th percentile value by
    axis.

    Parameters
    ----------
    curr_features : pd.DataFrame
        The current features that have been already computed.

    no_noise_acc : np.array
        The accelerometer data from a single participant with noise removed.

    axes_map : Dict of int, optional
        Denotes if the passed in no_noise_acc data is the vector magnitude.
        Defaults to using tri-axial accelerometer data.

        * Keys : string
            The given axis.

        * Values : int
            The given axis as an integer.

    Returns
    -------
    curr_features : pd.DataFrame
        The features that have been already computed plus the additional
        25th percentile features.
    """

    if len(axes_map) == 1:
        temp_df = pd.DataFrame(no_noise_acc.T)
        percentile25_df = temp_df.apply(lambda x: np.percentile(x, 25))
        curr_features["ACC_percentile25_vm"] = percentile25_df
    else:
        for axis in axes_map:
            temp_df = pd.DataFrame(no_noise_acc[:, :, axes_map[axis]].T)
            percentile25_df = temp_df.apply(lambda x: np.percentile(x, 25))
            curr_features[f"ACC_percentile25_{axis}"] = percentile25_df
    return curr_features


def compute_50(curr_features, no_noise_acc, axes_map={"x": 0, "y": 1, "z": 2}):
    """
    From the given windowed accelerometer data, without DC
    component or high frequency noise, compute the 50th percentile (median)
    value by axis.

    Parameters
    ----------
    curr_features : pd.DataFrame
        The current features that have been already computed.

    no_noise_acc : np.array
        The accelerometer data from a single participant with noise removed.

    axes_map : Dict of int, optional
        Denotes if the passed in no_noise_acc data is the vector magnitude.
        Defaults to using tri-axial accelerometer data.

        * Keys : string
            The given axis.

        * Values : int
            The given axis as an integer.

    Returns
    -------
    curr_features : pd.DataFrame
        The features that have been already computed plus the additional
        50th percentile (median) features.
    """

    if len(axes_map) == 1:
        temp_df = pd.DataFrame(no_noise_acc.T)
        percentile50_df = temp_df.apply(lambda x: np.percentile(x, 50))
        curr_features["ACC_percentile50_vm"] = percentile50_df
    else:
        for axis in axes_map:
            temp_df = pd.DataFrame(no_noise_acc[:, :, axes_map[axis]].T)
            percentile50_df = temp_df.apply(lambda x: np.percentile(x, 50))
            curr_features[f"ACC_percentile50_{axis}"] = percentile50_df
    return curr_features


def compute_75(curr_features, no_noise_acc, axes_map={"x": 0, "y": 1, "z": 2}):
    """
    From the given windowed accelerometer data, without DC
    component or high frequency noise, compute the 75th percentile value by
    axis.

    Parameters
    ----------
    curr_features : pd.DataFrame
        The current features that have been already computed.

    no_noise_acc : np.array
        The accelerometer data from a single participant with noise removed.

    axes_map : Dict of int, optional
        Denotes if the passed in no_noise_acc data is the vector magnitude.
        Defaults to using tri-axial accelerometer data.

        * Keys : string
            The given axis.

        * Values : int
            The given axis as an integer.

    Returns
    -------
    curr_features : pd.DataFrame
        The features that have been already computed plus the additional
        75th percentile features.
    """

    if len(axes_map) == 1:
        temp_df = pd.DataFrame(no_noise_acc.T)
        percentile75_df = temp_df.apply(lambda x: np.percentile(x, 75))
        curr_features["ACC_percentile75_vm"] = percentile75_df
    else:
        for axis in axes_map:
            temp_df = pd.DataFrame(no_noise_acc[:, :, axes_map[axis]].T)
            percentile75_df = temp_df.apply(lambda x: np.percentile(x, 75))
            curr_features[f"ACC_percentile75_{axis}"] = percentile75_df
    return curr_features


def corr_2(
    curr_features,
    no_noise_acc,
    dir_1,
    dir_2,
    axes_map={"x": 0, "y": 1, "z": 2}
):
    """
    From the given windowed accelerometer data, without DC
    component or high frequency noise, compute the correlation of two axes.

    Parameters
    ----------
    curr_features : pd.DataFrame
        The current features that have been already computed.

    no_noise_acc : np.array
        The accelerometer data from a single participant with noise removed.

    dir_1 : string
        The first axis of acceleration to be considered.

    dir_2 : string
        The second axis of acceleration to be considered.

    axes_map : Dict of int, optional
        Denotes if the passed in no_noise_acc data is the vector magnitude.
        Defaults to using tri-axial accelerometer data.

        * Keys : string
            The given axis.

        * Values : int
            The given axis as an integer.

    Returns
    -------
    curr_features : pd.DataFrame
        The features that have been already computed plus the additional
        correlation feature.
    """

    axis_1 = axes_map[dir_1]
    axis_2 = axes_map[dir_2]
    curr_features[f"ACC_corr_{dir_1}{dir_2}"] = [
        np.corrcoef(window[:, axis_1], window[:, axis_2])[0, 1]
        for window in no_noise_acc
    ]
    return curr_features


def compute_zero_crossings(
    curr_features,
    no_noise_acc,
    axes_map={"x": 0, "y": 1, "z": 2}
):
    """
    From the given windowed accelerometer data, without DC
    component or high frequency noise, compute the number of zero-crossings by
    axis.

    Parameters
    ----------
    curr_features : pd.DataFrame
        The current features that have been already computed.

    no_noise_acc : np.array
        The accelerometer data from a single participant with noise removed.

    axes_map : Dict of int, optional
        Denotes if the passed in no_noise_acc data is the vector magnitude.
        Defaults to using tri-axial accelerometer data.

        * Keys : string
            The given axis.

        * Values : int
            The given axis as an integer.

    Returns
    -------
    curr_features : pd.DataFrame
        The features that have been already computed plus the additional
        zero-crossing features.
    """

    for axis in axes_map:
        curr_features[f"ACC_zerocross_{axis}"] = pd.DataFrame(
            no_noise_acc[:, :, axes_map[axis]].T
        ).apply(lambda x: len(np.where(np.diff(np.sign(x)))[0]))
    return curr_features


def compute_energy(
    curr_features,
    no_noise_acc,
    axes_map={"x": 0, "y": 1, "z": 2}
):
    """
    From the given windowed accelerometer data, without DC
    component or high frequency noise, compute the energy by axis.

    Parameters
    ----------
    curr_features : pd.DataFrame
        The current features that have been already computed.

    no_noise_acc : np.array
        The accelerometer data from a single participant with noise removed.

    axes_map : Dict of int, optional
        Denotes if the passed in no_noise_acc data is the vector magnitude.
        Defaults to using tri-axial accelerometer data.

        * Keys : string
            The given axis.

        * Values : int
            The given axis as an integer.

    Returns
    -------
    curr_features : pd.DataFrame
        The features that have been already computed plus the additional
        energy features.
    """

    if len(axes_map) == 1:
        temp_df = pd.DataFrame(no_noise_acc.T)
        curr_features["ACC_energy_vm"] = temp_df.apply(lambda x: np.sum(x**2))
    else:
        for axis in axes_map:
            temp_df = pd.DataFrame(no_noise_acc[:, :, axes_map[axis]].T)
            energy_df = temp_df.apply(lambda x: np.sum(x**2))
            curr_features[f"ACC_energy_{axis}"] = energy_df
    return curr_features


def compute_frequency_features(curr_features, no_noise_acc, vm=False):
    """
    From the given windowed, accelerometer data, without DC
    component or high frequency noise, compute the dominant frequency,
    dominant magnitude, and entropy from the FFT.

    Parameters
    ----------
    curr_features : pd.DataFrame
        The current features that have been already computed.

    no_noise_acc : np.array
        The accelerometer data from a single participant with noise removed.

    vm : bool, optional
        Denotes if the passed in no_noise_acc data is the vector magnitude.

    Returns
    -------
    curr_features : pd.DataFrame
        The features that have been already computed plus the additional
        frequency features.
    """

    features = ["dominantFr", "dominantMag", "entropy"]
    features_by_axis = {}

    if vm:
        axes_map = {"vm": "vm"}
    else:
        axes_map = {"x": 0, "y": 1, "z": 2}

    for axis in axes_map.keys():
        features_by_axis[axis] = {}

        if vm:
            axis_acc = no_noise_acc
        else:
            axis_acc = no_noise_acc[:, :, axes_map[axis]]

        freqs = fftfreq(int(config["FREQ"] * config["T"]), 1 / config["FREQ"])
        freqs = freqs[1 : int(config["FREQ"] * config["T"]) // 2]  # skip DC = 0

        fft_vals = fft(axis_acc, axis=1)

        pos_fft = np.abs(fft_vals[:, 1 : (axis_acc.shape[1]) // 2])  # skip DC

        # Get dominant freq.
        max_indices = np.argmax(pos_fft, axis=1)
        features_by_axis[axis]["dominantFr"] = freqs[max_indices]

        # Get magnitude at dominant frequency.
        magnitude = pos_fft
        features_by_axis[axis]["dominantMag"] = magnitude[
            np.arange(magnitude.shape[0]), max_indices
        ]

        # Compute entropy.
        power_spectrum = (
            np.take(pos_fft**2, indices=np.arange((pos_fft).shape[1]), axis=1)
            / np.sum(pos_fft**2, axis=1)[:, np.newaxis]
        )
        features_by_axis[axis]["entropy"] = -np.sum(
            power_spectrum * np.log2(power_spectrum), axis=1
        )

    for feature in features:
        for axis in axes_map:
            axis_feature_df = features_by_axis[axis][feature]
            curr_features[f"ACC_{feature}_{axis}"] = axis_feature_df

    return curr_features


def butter_low_windows(no_dc_data, order=4):
    """
    From the given windowed single-axis, accelerometer data, without DC
    component, remove high frequency noise.

    Parameters
    ----------
    no_dc_data : np.array
        The windowed accelerometer data for a single participant.

    order : int, optional
        The order of the filter. Defaults to 4.

    Returns
    -------
    np.array
        The windowed accelerometer data for a single axis without high
        frequency noise or the DC component.
    """

    HIGH_CUTOFF = config["FREQ"] / 2 - 1
    normal_cutoff = HIGH_CUTOFF / (config["FREQ"] / 2)
    sos = butter(order, normal_cutoff, btype="low", analog=False, output="sos")

    return np.array([sosfiltfilt(sos, window) for window in no_dc_data])


def remove_noise(windowed_accel):
    """
    From the given windowed accelerometer data, remove high frequency noise and
    the DC component.

    Parameters
    ----------
    windowed_accel : np.array
        The windowed accelerometer data for a single participant.

    Returns
    -------
    no_noise_accel : np.array
        The windowed accelerometer data for a single participant without high
        frequency noise or the DC component.
    """

    no_noise_accel = windowed_accel.copy()

    for axis in range(windowed_accel.shape[2]):
        axis_acc = windowed_accel[:, :, axis]
        # Remove DC component.
        acc_min_mean = axis_acc - np.mean(axis_acc, axis=1, keepdims=True)

        no_noise_accel[:, :, axis] = butter_low_windows(acc_min_mean)

    return no_noise_accel


def make_features(windowed_accel):
    """
    From the given windowed accelerometer data, compute all the features needed
    in the benchmarking experiments.

    Parameters
    ----------
    windowed_accel : np.array
        The windowed accelerometer data for a single participant.

    Returns
    -------
    curr_features : pd.DataFrame
        The computed features for each window of accelerometer data.
    """

    curr_features = pd.DataFrame()

    no_noise_accel = remove_noise(windowed_accel)

    curr_features = compute_min(curr_features, no_noise_accel)
    curr_features = compute_max(curr_features, no_noise_accel)
    curr_features = compute_mean(curr_features, no_noise_accel)
    curr_features = compute_var(curr_features, no_noise_accel)
    curr_features = compute_std(curr_features, no_noise_accel)
    curr_features = compute_skew(curr_features, no_noise_accel)
    curr_features = compute_kurt(curr_features, no_noise_accel)
    curr_features = compute_25(curr_features, no_noise_accel)
    curr_features = compute_50(curr_features, no_noise_accel)
    curr_features = compute_75(curr_features, no_noise_accel)
    curr_features = compute_energy(curr_features, no_noise_accel)

    curr_features = corr_2(curr_features, no_noise_accel, "x", "y")
    curr_features = corr_2(curr_features, no_noise_accel, "y", "z")
    curr_features = corr_2(curr_features, no_noise_accel, "x", "z")

    curr_features = compute_zero_crossings(curr_features, no_noise_accel)
    compute_frequency_features(curr_features, no_noise_accel)

    # Get the vector magnitude.
    vm = np.sqrt(
        no_noise_accel[:, :, 0] ** 2
        + no_noise_accel[:, :, 1] ** 2
        + no_noise_accel[:, :, 2] ** 2
    )

    curr_features = compute_min(curr_features, vm, {"vm": "vm"})
    curr_features = compute_max(curr_features, vm, {"vm": "vm"})
    curr_features = compute_mean(curr_features, vm, {"vm": "vm"})
    curr_features = compute_var(curr_features, vm, {"vm": "vm"})
    curr_features = compute_std(curr_features, vm, {"vm": "vm"})
    curr_features = compute_skew(curr_features, vm, {"vm": "vm"})
    curr_features = compute_kurt(curr_features, vm, {"vm": "vm"})
    curr_features = compute_25(curr_features, vm, {"vm": "vm"})
    curr_features = compute_50(curr_features, vm, {"vm": "vm"})
    curr_features = compute_75(curr_features, vm, {"vm": "vm"})
    curr_features = compute_energy(curr_features, vm, {"vm": "vm"})

    return curr_features
