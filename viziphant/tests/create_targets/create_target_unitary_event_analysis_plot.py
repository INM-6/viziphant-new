import os

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq

import elephant.unitary_event_analysis as ue
from viziphant.unitary_event_analysis_plot import \
    plot_unitary_event_full_analysis, plot_spike_events, plot_spike_rates, \
    plot_coincidence_events, plot_coincidence_rates, \
    plot_statistical_significance, plot_unitary_events


target_images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "target_images")
PLOT_UE_FULL_ANALYSIS_TARGET_PATH = os.path.join(
    target_images_dir, "target_plot_UE_full_analysis.png")
PLOT_SPIKE_EVENTS_TARGET_PATH = os.path.join(
    target_images_dir, "target_plot_spike_events.png")
PLOT_SPIKE_RATES_TARGET_PATH = os.path.join(
    target_images_dir, "target_plot_spike_rates.png")
PLOT_COINCIDENCE_EVENTS_TARGET_PATH = os.path.join(
    target_images_dir, "target_plot_coincidence_events.png")
PLOT_COINCIDENCE_RATES_TARGET_PATH = os.path.join(
    target_images_dir, "target_plot_coincidence_rates.png")
PLOT_STATISTICAL_SIGNIFICANCE_TARGET_PATH = os.path.join(
    target_images_dir, "target_plot_statistical_significance.png")
PLOT_UNITARY_EVENTS_TARGET_PATH = os.path.join(
    target_images_dir, "target_plot_unitary_events.png")


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


def create_target_plot_ue_full_analysis():
    target_image_plot_ue_full_analysis = plt.figure("1", figsize=(20, 20))
    plot_unitary_event_full_analysis(
        SPIKETRAINS, UE, ue.jointJ(0.05), binsize=5 * pq.ms,
        window_size=100 * pq.ms, window_step=10 * pq.ms, n_neurons=2,
        position=((6, 1, 1), (6, 1, 2), (6, 1, 3), (6, 1, 4), (6, 1, 5),
                  (6, 1, 6)), plot_params_and_markers_user={'fsize': 14})
    target_image_plot_ue_full_analysis.savefig(
        PLOT_UE_FULL_ANALYSIS_TARGET_PATH)


def create_target_spike_events():
    target_image_plot_spike_events = plt.figure("2", figsize=(20, 20))
    plot_spike_events(
        SPIKETRAINS, window_size=100 * pq.ms, window_step=10 * pq.ms,
        n_neurons=2, position=(1, 1, 1),
        plot_params_and_markers_user={'fsize': 14})
    target_image_plot_spike_events.savefig(PLOT_SPIKE_EVENTS_TARGET_PATH)


def create_target_spike_rates():
    target_image_plot_spike_rates = plt.figure("3", figsize=(20, 20))
    plot_spike_rates(
        SPIKETRAINS, UE, window_size=100 * pq.ms,
        window_step=10 * pq.ms, n_neurons=2,
        position=(1, 1, 1), plot_params_and_marker_user={'fsize': 14})
    target_image_plot_spike_rates.savefig(PLOT_SPIKE_RATES_TARGET_PATH)


def create_target_coincidence_events():
    target_image_plot_coincidence_events = plt.figure(
        "4", figsize=(20, 20))
    plot_coincidence_events(
        SPIKETRAINS, UE, ue.jointJ(0.05), window_size=100 * pq.ms,
        window_step=10 * pq.ms, n_neurons=2, position=(1, 1, 1),
        plot_params_and_markers_user={'fsize': 14})
    target_image_plot_coincidence_events.\
        savefig(PLOT_COINCIDENCE_EVENTS_TARGET_PATH)


def create_target_coincidence_rates():
    target_image_plot_coincidences_rates = plt.figure(
        "5", figsize=(20, 20))
    plot_coincidence_rates(
        SPIKETRAINS, UE, window_size=100 * pq.ms,
        window_step=10 * pq.ms, n_neurons=2, position=(1, 1, 1),
        plot_params_and_markers_user={'fsize': 14})
    target_image_plot_coincidences_rates.savefig(
        PLOT_COINCIDENCE_RATES_TARGET_PATH)


def create_target_statistical_significance():
    target_image_plot_statistical_significance = plt.figure(
        "6", figsize=(20, 20))
    plot_statistical_significance(
        SPIKETRAINS, UE, ue.jointJ(0.05), window_size=100 * pq.ms,
        window_step=10 * pq.ms, n_neurons=2, position=(1, 1, 1),
        plot_params_and_markers_user={'fsize': 14})
    target_image_plot_statistical_significance.savefig(
        PLOT_STATISTICAL_SIGNIFICANCE_TARGET_PATH)


def create_target_unitary_events():
    target_image_plot_unitary_events = plt.figure("7", figsize=(20, 20))
    plot_unitary_events(
        SPIKETRAINS, UE, ue.jointJ(0.05), binsize=5 * pq.ms,
        window_size=100 * pq.ms, window_step=10 * pq.ms, n_neurons=2,
        position=(1, 1, 1), plot_params_and_markers_user={'fsize': 14})
    target_image_plot_unitary_events.savefig(
        PLOT_UNITARY_EVENTS_TARGET_PATH)

