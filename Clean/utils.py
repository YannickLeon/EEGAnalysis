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

from enum import Enum
class Behaviour(Enum):
    STATUS = "STATUS"
    SHOOT_BUTTON = "SHOOT_BUTTON"
    MISSILE_HIT_ENEMY = "MISSILE_HIT_ENEMY"
    COLLECT_STAR = "COLLECT_STAR"
    PLAYER_CRASH_ENEMY = "PLAYER_CRASH_ENEMY"
    PLAYER_CRASH_WALL = "PLAYER_CRASH_WALL"
    COLLECT_AMMO = "COLLECT_AMMO"

class Baseline(Enum):
    EXCLUDED = "EXCLUDED"
    INCLUDED = "INCLUDED"
    NAIVE = "NAIVE"

def plot_seperated(aggregated, not_aggregated, filename, title):
    plt.figure()
    plt.title(title)
    color_map = cm.get_cmap('tab10', len(not_aggregated))
    for i, deviation in enumerate(not_aggregated):
        plt.plot([j for j in range(8)], deviation, color=color_map(i))
    plt.plot([j for j in range(8)], aggregated, color="black", linewidth=3, linestyle='dashed')
    plt.savefig(filename)
    plt.close()

def plot_aggregated(aggregated, filename, title, labels=None, confidence_interval=None):
    plt.figure()
    plt.title(title)
    color_map = cm.get_cmap('tab10', len(aggregated))
    for i, baseline in enumerate(aggregated):
        x_vals = np.arange(len(aggregated[baseline]))
        color = color_map(i)
        if labels is None:
            plt.plot(x_vals, aggregated[baseline], color=color)
        else:
            plt.plot(x_vals, aggregated[baseline], color=color, label=labels[i])
        if confidence_interval is not None:
            lower, upper = confidence_interval
            plt.fill_between(x_vals, lower, upper, color=color, alpha=0.2)
    if labels is not None:
        plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_all(aggregated, not_aggregated, folder_path):
    def confidence(baseline):
        return get_confidence_intervals(not_aggregated[baseline], confidence=0.95)
    os.makedirs(folder_path, exist_ok=True)
    plot_seperated(aggregated[Baseline.EXCLUDED], not_aggregated[Baseline.EXCLUDED], folder_path + "/seperated_excluded.png", "Seperated subjects, excluded baseline")
    plot_seperated(aggregated[Baseline.INCLUDED], not_aggregated[Baseline.INCLUDED], folder_path + "/seperated_included.png", "Seperated subjects, included baseline")
    plot_seperated(aggregated[Baseline.NAIVE], not_aggregated[Baseline.NAIVE], folder_path + "/seperated_naive.png", "Seperated subjects, naive baseline")
    plot_aggregated(aggregated, folder_path + "/aggregated.png", "Aggegrated subjects with 3 different baselines", labels=["Excluded", "Included", "Naive"])
    plot_aggregated({Baseline.EXCLUDED: aggregated[Baseline.EXCLUDED]}, folder_path + "/aggregated_excluded.png", "Aggregated subjects, excluded baseline", confidence_interval=confidence(Baseline.EXCLUDED))
    plot_aggregated({Baseline.INCLUDED: aggregated[Baseline.INCLUDED]}, folder_path + "/aggregated_included.png", "Aggregated subjects, included baseline", confidence_interval=confidence(Baseline.INCLUDED))
    plot_aggregated({Baseline.NAIVE: aggregated[Baseline.NAIVE]}, folder_path + "/aggregated_naive.png", "Aggregated subjects, naive baseline", confidence_interval=confidence(Baseline.NAIVE))

def get_confidence_intervals(data, confidence):
    n = len(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standard_error = std / np.sqrt(n)
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin_of_error = t_critical * standard_error
    return mean - margin_of_error, mean + margin_of_error