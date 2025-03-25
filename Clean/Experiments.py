import os
from mne_bids import (BIDSPath,read_raw_bids)
import mne_bids
import mne
import importlib
import ccs_eeg_utils
from IPython.display import clear_output
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats
from utils import *
from tqdm import tqdm
from Preprocessing import Preprocessing

def run(combinations, sampling_rate=500, window=1, num_bins=8):
    """loads data for current channel and initiates processing"""
    prev_channel = -1
    data = {}
    bids_root = "../data/"
    preprocessing = Preprocessing(bids_root, "cache")

    peak_to_troughs = {}
    for behavior in Behaviour:
        peak_to_troughs[behavior] = {}
    for combination in tqdm(combinations):
        channel, behaviour = combination
        # load data for next channel
        if channel != prev_channel:
            data = preprocessing.load_channel(channel)
        peak_to_trough_raw, peak_to_trough_ica = process_combination(combination, data, sampling_rate, window, num_bins)
        peak_to_troughs[behaviour][channel] = (peak_to_trough_raw, peak_to_trough_ica)
    return peak_to_troughs

#  data: dict{subject: tuple(list[raw angles, ica angles], events)} data only contains information about a single channel
def process_combination(combination, data, sampling_rate, window, num_bins):
    channel, behavior = combination
    behavior_str = behavior.value if isinstance(behavior, Behaviour) else behavior
    results = get_deviations_for_subjects(data, data.keys(), behavior_str, sampling_rate, window, num_bins)
    aggregated_raw, not_aggregated_raw = results[0]
    aggregated_ica, not_aggregated_ica = results[1]
    plot_all(aggregated_raw, not_aggregated_raw, f"../Results/{channel}/{behavior_str}_raw")
    plot_all(aggregated_ica, not_aggregated_ica, f"../Results/{channel}/{behavior_str}_ica")
    peak_to_trough_raw = calculate_peak_to_trough(aggregated_raw)
    peak_to_trough_ica = calculate_peak_to_trough(aggregated_ica)
    return peak_to_trough_raw, peak_to_trough_ica

# data: dict{subject: tuple(list[raw angles, ica angles], events)} data only contains information about a single channel
def get_deviations_for_subjects(data, subjects, behavior, sampling_rate, window, num_bins):
    """
    compute deviations for a given channel over all subjects
    return: deviations (aggregated over subjects and seperate) for raw and ica
    """
    results = []
    for i in range(2):  # run analysis twice for raw angles and ica angles
        count = {}
        deviations = {}
        deviation_aggregated = {}
        for baseline in Baseline:
            deviations[baseline] = []
            deviation_aggregated[baseline] = np.zeros(num_bins)
            count[baseline] = 0

        for subject in subjects:
            angles, events = data[subject]
            angles = angles[i]
            filtered_events = events[events["trial_type"] == behavior]
            for baseline in Baseline:
                curr_deviation = get_bin_deviation(angles, [entry for entry in filtered_events["onset"]], baseline, timespan=window*sampling_rate, num_bins=num_bins)
                curr_deviation = np.array(curr_deviation)
                deviations[baseline].append(curr_deviation)
                weight = len(filtered_events)
                deviation_aggregated[baseline] += curr_deviation * weight
                count[baseline] += weight

        for baseline in Baseline:
            deviation_aggregated[baseline] /= count[baseline]
        results.append((deviation_aggregated, deviations))
    return results

# data: list[float] eeg-data for channel
def get_bin_deviation(data, timestamps, baseline_method, timespan = 1, num_bins = 8):
    """
    get bin distribution for selected baseline and compare to event-specific distribution
    return: deviation for each bin
    """
    if baseline_method == Baseline.EXCLUDED:
        base_data = list(data[0:int(timestamps[0]-timespan/2)])
        for i in range(len(timestamps)-1):
            base_data.extend(data[int((timestamps[i]+timespan/2)):int((timestamps[i+1]-timespan/2))])
        base_data.extend(data[int((timestamps[len(timestamps)-1]+timespan/2)):len(data)])
    if baseline_method == Baseline.INCLUDED:
        base_data = data

    if baseline_method == Baseline.NAIVE:
        default = [0.125]*8
    else:
        bins = create_bins(base_data, num_bins)
        default = get_bin_probabilities(bins)

    combined_data = []
    for timestamp in timestamps:
        combined_data.extend(data[int((timestamp-timespan/2)):int((timestamp+timespan/2))])

    event_bins = create_bins(combined_data, num_bins)
    event_probability = get_bin_probabilities(event_bins)

    deviation = []
    for i in range(len(default)):
        deviation.append(100*(event_probability[i]/default[i] - 1))

    return deviation

# data: list[float] eeg-data for channel (of event-neighbourhoods)
def create_bins(data, num_bins=8):
    """
    sort data into bins
    """
    bin_size = 2 * np.pi / num_bins
    data = np.array(data)
    bin_indices = np.floor((data + np.pi) / bin_size)
    bins = [data[bin_indices == i] for i in range(num_bins)]
    return bins

# bins: list[list[data for bin]*num_bins] list with data for each bin
def get_bin_probabilities(bins):
    """
    calculate relative frequency of bins
    """
    bin_sizes = np.array([len(bin) for bin in bins])
    total_size = np.sum(bin_sizes)
    return bin_sizes / total_size

# bins_by_basline: dict{Baseline: list[float*num_bins]} dict of bin probabilities for each basline
def calculate_peak_to_trough(bins_by_baseline):
    """
    calculates peak-to-through differences for each baseline
    """
    differences = {Baseline.EXCLUDED: None, Baseline.INCLUDED: None, Baseline.NAIVE: None}
    for key, bin in bins_by_baseline.items():
        min = np.min(bin)
        max = np.max(bin)
        differences[key] = max - min
    return differences