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
import pickle

class Preprocessing:
    def __init__(self, dataset_path, cache_path):
        self.dataset_path = dataset_path
        self.cache_path = cache_path
        os.makedirs(cache_path, exist_ok=True) 

    # store data for each channel: dict{channel, dict{subject, tuple(angles, events)}}
    def load_all_subjects(self):
        data = {}
        for i in range(1, 18):
            subject = str(i).zfill(3)
            angles = self.load_subject(subject)
            events = self.load_events(subject)
            data[subject] = (angles, events)
        return data

    def load_channel(self, channel):
        cache_file = os.path.join(self.cache_path, f"{str(channel).zfill(3)}_data.pkl")

        # Load from cache if available
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        
        # bischen reudig aber egal
        return self.prepare_channels()[channel]
        

    def prepare_channels(self):
        data = {}
        for i in range(65):
            data[i] = {}
            
        for s in range(1,18):
            subject = str(s).zfill(3)
            angles = self.load_subject(subject)
            events = self.load_events(subject)
            for i in range(len(angles[0])):
                data[i][subject] = ([angles[0][i], angles[1][i]], events)
                
        for i in range(len(data)):
            cache_file = os.path.join(self.cache_path, f"{str(i).zfill(3)}_data.pkl")
            if os.path.exists(cache_file):
                continue
            with open(cache_file, "wb") as f:
                pickle.dump(data[i], f)
        return data


    def load_subject(self, subject):
        cache_file = os.path.join(self.cache_path, f"{subject}_angles.pkl")

        # Load from cache if available
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        # Otherwise, process
        bids_path = BIDSPath(subject=subject, run="02", task="ContinuousVideoGamePlay",
                            datatype='eeg', suffix='eeg',
                            root=self.dataset_path)
        raw = read_raw_bids(bids_path)
        ccs_eeg_utils.read_annotations_core(bids_path,raw)
        angles = self.run_pipeline(raw)

        # And save computed results
        with open(cache_file, "wb") as f:
            pickle.dump(angles, f)

        return angles

    def run_pipeline(self, raw):
        raw.load_data()
        self.fix_channel_types(raw)
        self.fix_montage(raw)
        raw_ica = self.apply_ica(raw)
        raws = [raw, raw_ica]
        angles = []
        for raw in raws:
            raw.filter(0.01, 0.1, picks="all")
            raw.apply_hilbert(picks="all")
            angles.append(np.angle(raw.get_data(picks="all")))
        return angles

    def fix_channel_types(self, raw):
        list_name = raw.ch_names
        list_type = ['eeg' if i < len(list_name)-2 else 'eog' for i in range(len(list_name))]
        raw.set_channel_types(dict(zip(list_name, list_type)))

    def fix_montage(self, raw):
        path = self.dataset_path + "sub-001/eeg/sub-001_task-ContinuousVideoGamePlay_run-02_electrodes.tsv"
        electrodes = pd.read_csv(path, sep='\t')
        original_montage = raw.get_montage()
        nasion = original_montage.get_positions()["nasion"]
        lpa = original_montage.get_positions()["lpa"]
        rpa = original_montage.get_positions()["rpa"]
        hsp = original_montage.get_positions()["hsp"]
        hpi = original_montage.get_positions()["hpi"]
        montage = mne.channels.make_dig_montage(ch_pos=dict(zip(electrodes['name'], electrodes[['x','y','z']].values)), 
                                                nasion=nasion, lpa=lpa, rpa=rpa, hsp=hsp, hpi=hpi)
        raw.set_montage(montage)

    def apply_ica(self, raw):
        raw_filtered = raw.copy()
        raw_filtered.filter(1, None, picks="all")
        ica = mne.preprocessing.ICA(n_components=30, method="picard")
        ica.fit(raw_filtered, picks="all", verbose=True)
        ica.exclude = [0, 1]
        raw_ica = raw.copy()
        ica.apply(raw_ica)
        return raw_ica        

    def load_events(self, subject):
        df = pd.read_csv(f'{self.dataset_path}sub-{subject}/eeg/sub-{subject}_task-ContinuousVideoGamePlay_run-02_events.tsv', sep='\t')
        return df