import tempfile
import unittest

import numpy as np
from viziphant.unitary_event_analysis_plot import \
    plot_unitary_event_full_analysis, plot_spike_events, plot_spike_rates, \
    plot_coincidence_events, plot_coincidence_rates, \
    plot_statistical_significance, plot_unitary_events
import viziphant.tests.create_targets.\
    create_target_unitary_event_analysis_plot as cti #creat target image
import quantities as pq
import neo
import elephant.unitary_event_analysis as ue
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def compare_images(path_target_image, path_result_image):
    target_image_as_array = mpimg.imread(path_target_image)
    result_image_as_array = mpimg.imread(path_result_image)
    the_same = (result_image_as_array == target_image_as_array).all()

    return the_same


class UnitaryEventAnalysisPlotTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # aus Jupyter-Notebook
        # Download data
        # url = 'https://github.com/INM-6/elephant-tutorial-data/raw/master/
        # dataset-1/dataset-1.h5'
        # test_dataset = wget.download(url)
        # Load data and extract spiketrains
        cls.block = neo.io.NeoHdf5IO('dataset-1.h5')
        cls.sts1 = cls.block.read_block().segments[0].spiketrains
        cls.sts2 = cls.block.read_block().segments[1].spiketrains
        cls.spiketrains = np.vstack((cls.sts1, cls.sts2)).T
        cls.UE = ue.jointJ_window_analysis(
            cls.spiketrains, binsize=5 * pq.ms, winsize=100 * pq.ms,
            winstep=10 * pq.ms, pattern_hash=[3])
        # TODO: once unitary_event_analysis is finished,
        # creat a fixed target image for each function

        # create 1.figure with 6 subplots with
        cti.create_target_plot_UE_full_analysis()
        cls.path_target_image_plot_UE_full_analysis = \
           cti.PLOT_UE_FULL_ANALYSIS_TARGET_PATH

        # create 2.figure with 1 subplot with plot_spike_events():
        cti.create_target_spike_events()
        cls.path_target_image_plot_spike_events = \
            cti.PLOT_SPIKE_EVENTS_TARGET_PATH

        # create 3.figure with 1 subplot with plot_spike_rates():
        cti.create_target_spike_rates()
        cls.path_target_image_plot_spike_rates = \
            cti.PLOT_SPIKE_RATES_TARGET_PATH

        # create 4.figure with 1 subplot with plot_coincidence_events():
        cti.create_target_coincidence_events()
        cls.path_target_image_plot_coincidence_events = \
            cti.PLOT_COINCIDENCE_EVENTS_TARGET_PATH

        # create 5.figure with 1 subplot with plot_coincidence_rates():
        cti.create_target_coincidence_rates()
        cls.path_target_image_plot_coincidence_rates = \
            cti.PLOT_COINCIDENCE_RATES_TARGET_PATH

        # create 6.figure with 1 subplot with plot_statistical_significance():
        cti.create_target_statistical_significance()
        cls.path_target_image_plot_statistical_significance = \
            cti.PLOT_STATISTICAL_SIGNIFICANCE_TARGET_PATH

        # create 7.figure with 1 subplot with plot_unitary_events():
        cti.create_target_unitary_events()
        cls.path_target_image_plot_unitary_events = \
            cti.PLOT_UNITARY_EVENTS_TARGET_PATH

    def test_plot_unitary_event_full_analysis(self):
        # create result image
        self.result_image_plot_UE_full_analysis = plt.figure("1.1",
                                                         figsize=(20, 20))
        plot_unitary_event_full_analysis(
            self.spiketrains, self.UE, ue.jointJ(0.05), binsize=5 * pq.ms,
            window_size=100 * pq.ms, window_step=10 * pq.ms, n_neurons=2,
            position=((6, 1, 1), (6, 1, 2), (6, 1, 3), (6, 1, 4), (6, 1, 5),
                      (6, 1, 6)), plot_params_and_markers_user={'fsize': 14})
        self.path_result_image_plot_UE_full_analysis = \
            tempfile.mkstemp(suffix=".png")[1]
        self.result_image_plot_UE_full_analysis.savefig(
            self.path_result_image_plot_UE_full_analysis)

        # assertion
        self.assertTrue(compare_images(
            self.path_target_image_plot_UE_full_analysis,
            self.path_result_image_plot_UE_full_analysis))

    def test_plot_spike_events(self):
        # create result image
        self.result_image_plot_spike_events = plt.figure("2.1",
                                                         figsize=(20, 20))
        plot_spike_events(
            self.spiketrains, window_size=100 * pq.ms, window_step=10 * pq.ms,
            n_neurons=2, position=(1, 1, 1),
            plot_params_and_markers_user={'fsize': 14})
        self.path_result_image_plot_spike_events = \
            tempfile.mkstemp(suffix=".png")[1]
        self.result_image_plot_spike_events.savefig(
            self.path_result_image_plot_spike_events)

        # assertion
        self.assertTrue(compare_images(
            self.path_target_image_plot_spike_events,
            self.path_result_image_plot_spike_events))

    def test_plot_spike_rates(self):
        # create result image
        self.result_image_plot_spike_rates = plt.figure("3.1",
                                                        figsize=(20, 20))
        plot_spike_rates(
            self.spiketrains, self.UE, window_size=100 * pq.ms,
            window_step=10 * pq.ms, n_neurons=2,
            position=(1, 1, 1), plot_params_and_marker_user={'fsize': 14})
        self.path_result_image_plot_spike_rates = \
            tempfile.mkstemp(suffix=".png")[1]
        self.result_image_plot_spike_rates.savefig(
            self.path_result_image_plot_spike_rates)

        # assertion
        self.assertTrue(compare_images(
            self.path_target_image_plot_spike_rates,
            self.path_result_image_plot_spike_rates))

    def test_plot_coincidence_events(self):
        # create result image
        self.result_image_plot_coincidence_events = plt.figure(
            "4.1", figsize=(20, 20))
        plot_coincidence_events(
            self.spiketrains, self.UE, ue.jointJ(0.05), window_size=100 * pq.ms,
            window_step=10 * pq.ms, n_neurons=2, position=(1, 1, 1),
            plot_params_and_markers_user={'fsize': 14})
        self.path_result_image_plot_coincidence_events = \
            tempfile.mkstemp(suffix=".png")[1]
        self.result_image_plot_coincidence_events.savefig(
            self.path_result_image_plot_coincidence_events)

        # assertion
        self.assertTrue(compare_images(
            self.path_target_image_plot_coincidence_events,
            self.path_result_image_plot_coincidence_events))

    def test_plot_coincidence_rates(self):
        # create result image
        self.result_image_plot_coincidences_rates = plt.figure(
            "5.1", figsize=(20, 20))
        plot_coincidence_rates(
            self.spiketrains, self.UE, window_size=100 * pq.ms,
            window_step=10 * pq.ms, n_neurons=2, position=(1, 1, 1),
            plot_params_and_markers_user={'fsize': 14})
        self.path_result_image_plot_coincidence_rates = \
            tempfile.mkstemp(suffix=".png")[1]
        self.result_image_plot_coincidences_rates.savefig(
            self.path_result_image_plot_coincidence_rates)

        # assertion
        self.assertTrue(compare_images(
            self.path_target_image_plot_coincidence_rates,
            self.path_result_image_plot_coincidence_rates))

    def test_plot_statistical_significance(self):
        # create result image
        self.result_image_plot_statistical_significance = plt.figure(
            "6.1", figsize=(20, 20))
        plot_statistical_significance(
            self.spiketrains, self.UE, ue.jointJ(0.05), window_size=100 * pq.ms,
            window_step=10 * pq.ms, n_neurons=2, position=(1, 1, 1),
            plot_params_and_markers_user={'fsize': 14})
        self.path_result_image_plot_statistical_significance = \
            tempfile.mkstemp(suffix=".png")[1]
        self.result_image_plot_statistical_significance.savefig(
            self.path_result_image_plot_statistical_significance)

        # assertion
        self.assertTrue(compare_images(
            self.path_target_image_plot_statistical_significance,
            self.path_result_image_plot_statistical_significance))

    def test_plot_unitary_events(self):
        # create result image
        self.result_image_plot_unitary_events = plt.figure("7.1",
                                                           figsize=(20, 20))
        plot_unitary_events(
            self.spiketrains, self.UE, ue.jointJ(0.05), binsize=5 * pq.ms,
            window_size=100 * pq.ms, window_step=10 * pq.ms, n_neurons=2,
            position=(1, 1, 1), plot_params_and_markers_user={'fsize': 14})
        self.path_result_image_plot_unitary_events = \
            tempfile.mkstemp(suffix=".png")[1]
        self.result_image_plot_unitary_events.savefig(
            self.path_result_image_plot_unitary_events)

        # assertion
        self.assertTrue(compare_images(
            self.path_target_image_plot_unitary_events,
            self.path_result_image_plot_unitary_events))


if __name__ == '__main__':
    unittest.main()
