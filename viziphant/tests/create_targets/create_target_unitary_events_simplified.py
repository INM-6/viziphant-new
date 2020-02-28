import os

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq

import elephant.unitary_event_analysis as ue
from viziphant.plot_unitary_events_simplified import \
    plot_unitary_events_simplified

target_images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "target_images")
PLOT_UNITARY_EVENTS_SIMPLIFIED_TARGET_PATH = os.path.join(
    target_images_dir, "target_plot_unitary_events_simplified.png")

# aus Jupyter-Notebook
# Download data
# url = 'https://github.com/INM-6/elephant-tutorial-data/raw/master/
# dataset-1/dataset-1.h5'
# test_dataset = wget.download(url)
# Load data and extract spiketrains
BLOCK = neo.io.NeoHdf5IO('dataset-1.h5')
STS1 = BLOCK.read_block().segments[0].spiketrains
STS2 = BLOCK.read_block().segments[1].spiketrains
SPIKETRAINS = np.vstack((STS1, STS2)).T
UE = ue.jointJ_window_analysis(
    SPIKETRAINS, binsize=5 * pq.ms, winsize=100 * pq.ms,
    winstep=10 * pq.ms, pattern_hash=[3])


def create_target_unitary_events_simplified():
    target_image_plot_unitary_events_simplified = plt.figure(
        "1", figsize=(20, 20))
    plot_unitary_events_simplified(
        SPIKETRAINS, UE, ue.jointJ(0.05), binsize=5 * pq.ms,
        window_size=100 * pq.ms, window_step=10 * pq.ms, n_neurons=2)
    target_image_plot_unitary_events_simplified.savefig(
        PLOT_UNITARY_EVENTS_SIMPLIFIED_TARGET_PATH)
