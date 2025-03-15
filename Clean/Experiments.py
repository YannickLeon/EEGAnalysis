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
import torch
import time

def run(data, combinations, baselines, sampling_rate, window, num_bins):
        for combination in tqdm(combinations):
            process_combination(combination, data, baselines, sampling_rate, window, num_bins)

def process_combination(combination, data, baselines, sampling_rate, window, num_bins):
    channel, behavior = combination
    behavior_str = behavior.value if isinstance(behavior, Behaviour) else behavior
    results = get_deviations_for_subjects(data, data.keys(), channel, behavior_str, baselines, sampling_rate, window, num_bins)
    aggregated_raw, not_aggregated_raw = results[0]
    aggregated_ica, not_aggregated_ica = results[1]
    plot_all(aggregated_raw, not_aggregated_raw, f"../Results/{channel}/{behavior_str}_raw")
    plot_all(aggregated_ica, not_aggregated_ica, f"../Results/{channel}/{behavior_str}_ica")

def get_deviations_for_subjects(data, subjects, channel, behavior, baselines, sampling_rate, window, num_bins):
    results = []
    for i in range(2):
        method_count = len(baselines)
        deviation_aggregated = np.zeros((method_count, num_bins))
        count = np.zeros(method_count)
        deviations = [[] for _ in range(method_count)]

        for subject in subjects:
            angles, events = data[subject]
            angles = angles[i]
            filtered_events = events[events["trial_type"] == behavior]
            data_of_channel = angles[range(channel, channel+1)]
            for method_idx, method in enumerate(baselines):
                curr_deviation = get_total_deviation(data_of_channel, [entry for entry in filtered_events["onset"]], method, sampling_rate=sampling_rate, timespan=window, num_bins=num_bins)
                curr_deviation = np.array(curr_deviation)
                deviations[method_idx].append(curr_deviation)
                weight = len(filtered_events)
                deviation_aggregated[method_idx] += curr_deviation * weight
                count[method_idx] += weight

        for j in range(method_count):
            deviation_aggregated[j] /= count[j]
        deviations = [np.array(method_devs) for method_devs in deviations]
        results.append((deviation_aggregated, deviations))
    return results
    
def get_total_deviation(data, timestamps, baseline_method, timespan = 1, sampling_rate = 500, num_bins = 8):
    total_deviation = [0 for i in range(num_bins)]
    for i in range(len(data)):
        deviation = get_bin_deviation(data[i], [timestamp*sampling_rate for timestamp in timestamps], baseline_method, timespan*sampling_rate, num_bins)
        for i in range(len(deviation)):
            total_deviation[i] += deviation[i]
    
    for i in range(len(total_deviation)):
        total_deviation[i] /= len(data)
    return total_deviation

def get_bin_deviation(data, timestamps, baseline_method, timespan = 1 * 500, num_bins = 8):
    if baseline_method == "exclude_event":
        base_data = list(data[0:int(timestamps[0]-timespan/2)])
        for i in range(len(timestamps)-1):
            base_data.extend(data[int((timestamps[i]+timespan/2)):int((timestamps[i+1]-timespan/2))])
        base_data.extend(data[int((timestamps[len(timestamps)-1]+timespan/2)):len(data)])
    if baseline_method == "include_event":
        base_data = data

    if baseline_method == "naive":
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
    
def create_bins(data, num_bins=8):
    bin_size = 2 * np.pi / num_bins
    data = np.array(data)
    bin_indices = np.floor((data + np.pi) / bin_size)
    bins = [data[bin_indices == i] for i in range(num_bins)]
    return bins

def get_bin_probabilities(bins):
    bin_sizes = np.array([len(bin) for bin in bins])
    total_size = np.sum(bin_sizes)
    return bin_sizes / total_size