from mne_bids import (BIDSPath,read_raw_bids)
import mne_bids
import mne
import importlib
import ccs_eeg_utils
from IPython.display import clear_output
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

behaviors = ["STATUS", "GAME OVER", "SHOOT_BUTTON", "MISSILE_HIT_ENEMY", "COLLECT_STAR", "PLAYER_CRASH_ENEMY", "PLAYER_CRASH_WALL", "COLLECT_AMMO"]

def filter_for_behavior(df, data, behavior, sampling_rate=500, window=100):
    filtered_df = df[df['trial_type'] == behavior]
    onset_times = filtered_df['onset'].values
    eeg_data_points = (onset_times * sampling_rate).astype(int)
    extracted_data = np.array([data[idx:idx + window] for idx in eeg_data_points if idx + window <= len(data)])
    return extracted_data

def create_bins(data, num_bins=8):
    bin_size = 2 * np.pi / num_bins
    bins = []
    for i in range(num_bins):
        bins.append([])
    for element in data:
        bin_index = int((element + np.pi) // bin_size)
        bins[bin_index].append(element)
    return bins

angles = []

for sub in range(1, 16):
    bids_root = "./data/"
    subject_id = str(sub).zfill(3)

    bids_path = BIDSPath(subject=subject_id, run="02", task="ContinuousVideoGamePlay",
                        datatype='eeg', suffix='eeg',
                        root=bids_root)

    # read the file
    raw = read_raw_bids(bids_path)
    # fix the annotations readin
    ccs_eeg_utils.read_annotations_core(bids_path,raw)
    clear_output()
    raw.load_data()
    raw.filter(0.01, 0.1, picks="all")
    raw.apply_hilbert(picks="all")
    test = np.angle(raw.get_data(picks="all"))

    df = pd.read_csv(f'data\\sub-{subject_id}\\eeg\\sub-{subject_id}_task-ContinuousVideoGamePlay_run-02_events.tsv', sep='\t')
    # for i in range(0, 65):
    i = 12
    res = filter_for_behavior(df, test[i], "COLLECT_AMMO")
    res = res.flatten()
    angles.append(res)

angles_flat = np.concatenate(angles)
bins = create_bins(angles_flat = np.concatenate(angles))
plt.bar(range(8), [len(bin) for bin in bins])