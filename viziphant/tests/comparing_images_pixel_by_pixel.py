
import unittest

import numpy as np
from viziphant.unitary_event_analysis_PLOT import \
    plot_unitary_event_full_analysis, plot_spike_events, plot_spike_rates, \
    plot_coincidence_events, plot_coincidence_rates, \
    plot_statistical_significance, plot_unitary_events
import quantities as pq
import neo
import elephant.unitary_event_analysis as ue
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def compare_images(self, path_target_image, path_result_image):
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
        # plot_unitary_event_full_analysis():
        cls.target_image1_plot_UE_full_analysis = plt.figure("1",
                                                             figsize=(20, 20))
        plot_unitary_event_full_analysis(
            cls.spiketrains, cls.UE, ue.jointJ(0.05), binsize=5 * pq.ms,
            window_size=100 * pq.ms, window_step=10 * pq.ms, n_neurons=2,
            plot_params_user={'fsize': 14},
            plot_markers_user=[{},
                               {'data_markercolor': ("r", "b")},
                               {},
                               {'data_markercolor': ("m", "g")},
                               {'data_markercolor': ("k", "b", "r")},
                               {}],
            position=((6, 1, 1), (6, 1, 2), (6, 1, 3), (6, 1, 4), (6, 1, 5),
                      (6, 1, 6)))
        import os
        print ("Standort2: ", os.getcwd())

        cls.target_image1_plot_UE_full_analysis.savefig(
            "/home/kramer/Documents/Studium Angewandte Mathematik und "
            "Informatik/INM6/PycharmProjects/viziphant-new/"
            "viziphant/tests/target_images/"
            "target_image1_plot_UE_full_analysis.png")
        cls.path_target_image1__plot_UE_full_analysis = "target_images/" \
            "target_image1_plot_UE_full_analysis.png"

        # create 2.figure with 1 subplot with plot_spike_events():
        cls.target_image2_plot_spike_events = plt.figure("2", figsize=(20, 20))
        plot_spike_events(
            cls.spiketrains, window_size=100 * pq.ms, window_step=10 * pq.ms,
            n_neurons=2, plot_params_user={'fsize': 14}, plot_markers_user=[],
            position=(1, 1, 1))
        cls.target_image2_plot_spike_events.savefig(
            "/home/kramer/Documents/Studium Angewandte Mathematik und "
            "Informatik/INM6/PycharmProjects/viziphant-new/viziphant/tests/"
            "target_images/target_image2_plot_spike_events.png")
        cls.path_target_image2_plot_spike_events = "target_images/" \
            "target_image2_plot_spike_events.png"

        # create 3.figure with 1 subplot with plot_spike_rates():
        cls.target_image3_plot_spike_rates = plt.figure("3", figsize=(20, 20))
        plot_spike_rates(
            cls.spiketrains, cls.UE, window_size=100 * pq.ms,
            window_step=10 * pq.ms, n_neurons=2,
            plot_params_user={'fsize': 14},
            plot_markers_user={'data_markercolor': ("r", "b")},
            position=(1, 1, 1))
        cls.target_image3_plot_spike_rates.savefig(
            "/home/kramer/Documents/Studium Angewandte Mathematik und "
            "Informatik/INM6/PycharmProjects/viziphant-new/viziphant/tests/"
            "target_images/target_image3_plot_spike_rates.png")
        cls.path_target_image3_plot_spike_rates = "target_images/" \
            "target_image3_plot_spike_rates.png"

        # create 4.figure with 1 subplot with plot_coincidence_events():
        cls.target_image4_plot_coincidence_events = plt.figure(
            "4", figsize=(20, 20))
        plot_coincidence_events(
            cls.spiketrains, cls.UE, ue.jointJ(0.05), window_size=100 * pq.ms,
            window_step=10 * pq.ms, n_neurons=2,
            plot_params_user={'fsize': 14}, plot_markers_user={},
            position=(1, 1, 1))
        cls.target_image4_plot_coincidence_events.savefig(
            "/home/kramer/Documents/Studium Angewandte Mathematik und "
            "Informatik/INM6/PycharmProjects/viziphant-new/viziphant/tests/"
            "target_images/target_image4_plot_coincidence_events.png")
        cls.path_target_image4_plot_coincidence_events = "target_images/" \
            "target_image4_plot_coincidence_events.png"

        # create 5.figure with 1 subplot with plot_coincidence_rates():
        cls.target_image5_plot_coincidences_rates = plt.figure(
            "5", figsize=(20, 20))
        plot_coincidence_rates(
            cls.spiketrains, cls.UE, window_size=100 * pq.ms,
            window_step=10 * pq.ms, n_neurons=2,
            plot_params_user={'fsize': 14},
            plot_markers_user={'data_markercolor': ("m", "g")},
            position=(1, 1, 1))
        cls.target_image5_plot_coincidences_rates.savefig(
            "/home/kramer/Documents/Studium Angewandte Mathematik und "
            "Informatik/INM6/PycharmProjects/viziphant-new/viziphant/tests/"
            "target_images/target_image5_plot_coincidence_rates.png")
        cls.path_target_image5_plot_coincidence_rates = "target_images/" \
            "target_image5_plot_coincidence_rates.png"

        # create 6.figure with 1 subplot with plot_statistical_significance():
        cls.target_image6_plot_statistical_significance = plt.figure(
            "6", figsize=(20, 20))
        plot_statistical_significance(
            cls.spiketrains, cls.UE, ue.jointJ(0.05), window_size=100 * pq.ms,
            window_step=10 * pq.ms, n_neurons=2,
            plot_params_user={'fsize': 14},
            plot_markers_user={'data_markercolor': ("k", "b", "r")},
            position=(1, 1, 1))
        cls.target_image6_plot_statistical_significance.savefig(
            "/home/kramer/Documents/Studium Angewandte Mathematik und "
            "Informatik/INM6/PycharmProjects/viziphant-new/viziphant/tests/"
            "target_images/target_image6_plot_statistical_significance.png")
        cls.path_target_image6_plot_statistical_significance = "target_images"\
            "/target_image6_plot_statistical_significance.png"

        # create 7.figure with 1 subplot with plot_unitary_events():
        cls.target_image7_plot_unitary_events = plt.figure("7",
                                                           figsize=(20, 20))
        plot_unitary_events(
            cls.spiketrains, cls.UE, ue.jointJ(0.05), binsize=5 * pq.ms,
            window_size=100 * pq.ms, window_step=10 * pq.ms, n_neurons=2,
            plot_params_user={'fsize': 14}, plot_markers_user={},
            position=(1, 1, 1))
        cls.target_image7_plot_unitary_events.savefig(
            "/home/kramer/Documents/Studium Angewandte Mathematik und "
            "Informatik/INM6/PycharmProjects/viziphant-new/viziphant/tests/"
            "target_images/target_image7_plot_unitary_events.png")
        cls.path_target_image7_plot_unitary_events = "target_images/" \
            "target_image7_plot_unitary_events.png"

    def test_plot_unitary_event_full_analysis(self):
        # create result image
        result_image1_plot_UE_full_analysis = plt.figure("1.1",
                                                         figsize=(20, 20))
        plot_unitary_event_full_analysis(
            self.spiketrains, self.UE, ue.jointJ(0.05), binsize=5 * pq.ms,
            window_size=100 * pq.ms, window_step=10 * pq.ms, n_neurons=2,
            plot_params_user={'fsize': 14},
            plot_markers_user=[{},
                               {'data_markercolor': ("r", "b")},
                               {},
                               {'data_markercolor': ("m", "g")},
                               {'data_markercolor': ("k", "b", "r")},
                               {}],
            position=((6, 1, 1), (6, 1, 2), (6, 1, 3), (6, 1, 4), (6, 1, 5),
                      (6, 1, 6)))
        result_image1_plot_UE_full_analysis.savefig(
            "/home/kramer/Documents/Studium Angewandte Mathematik und "
            "Informatik/INM6/PycharmProjects/viziphant-new/viziphant/tests/"
            "target_images/result_image1_plot_UE_full_analysis.png")
        self.path_result_image1__plot_UE_full_analysis = "target_images/" \
            "result_image1_plot_UE_full_analysis.png"

        # assertion
        self.assertTrue(compare_images(
            self, self.path_target_image1__plot_UE_full_analysis,
            self.path_result_image1__plot_UE_full_analysis))

    def test_plot_spike_events(self):
        # creat result image
        self.result_image2_plot_spike_events = plt.figure("2.1",
                                                          figsize=(20, 20))
        plot_spike_events(
            self.spiketrains, window_size=100 * pq.ms, window_step=10 * pq.ms,
            n_neurons=2, plot_params_user={'fsize': 14}, plot_markers_user=[],
            position=(1, 1, 1))
        self.result_image2_plot_spike_events.savefig(
            "/home/kramer/Documents/Studium Angewandte Mathematik und "
            "Informatik/INM6/PycharmProjects/viziphant-new/viziphant/tests/"
            "target_images/result_image2_plot_spike_events.png")
        self.path_result_image2_plot_spike_events = "target_images/" \
            "result_image2_plot_spike_events.png"

        # import os
        # print ("Standort2: " , os.getcwd())
        #
        # assertion
        self.assertTrue(compare_images(
            self, self.path_target_image2_plot_spike_events,
            self.path_result_image2_plot_spike_events))

    def test_plot_spike_rates(self):
        # create result image
        self.result_image3_plot_spike_rates = plt.figure("3.1",
                                                         figsize=(20, 20))
        plot_spike_rates(
            self.spiketrains, self.UE, window_size=100 * pq.ms,
            window_step=10 * pq.ms, n_neurons=2,
            plot_params_user={'fsize': 14},
            plot_markers_user={'data_markercolor': ("r", "b")},
            position=(1, 1, 1))
        self.result_image3_plot_spike_rates.savefig(
            "/home/kramer/Documents/Studium Angewandte Mathematik und "
            "Informatik/INM6/PycharmProjects/viziphant-new/viziphant/tests/"
            "target_images/result_image3_plot_spike_rates.png")
        self.path_result_image3_plot_spike_rates = "target_images/" \
            "result_image3_plot_spike_rates.png"

        # assertion
        self.assertTrue(compare_images(
            self, self.path_target_image3_plot_spike_rates,
            self.path_result_image3_plot_spike_rates))

    def test_plot_coincidence_events(self):
        # create result image
        self.result_image4_plot_coincidence_events = plt.figure(
            "4.1", figsize=(20, 20))
        plot_coincidence_events(
            self.spiketrains, self.UE, ue.jointJ(0.05),
            window_size=100 * pq.ms, window_step=10 * pq.ms, n_neurons=2,
            plot_params_user={'fsize': 14}, plot_markers_user={},
            position=(1, 1, 1))
        self.result_image4_plot_coincidence_events.savefig(
            "/home/kramer/Documents/Studium Angewandte Mathematik und "
            "Informatik/INM6/PycharmProjects/viziphant-new/viziphant/tests/"
            "target_images/result_image4_plot_coincidence_events.png")
        self.path_result_image4_plot_coincidence_events = "target_images/" \
            "result_image4_plot_coincidence_events.png"

        # import os
        # print ("standort4: ", os.getcwd())
        # assertion
        self.assertTrue(compare_images(
            self, self.path_target_image4_plot_coincidence_events,
            self.path_result_image4_plot_coincidence_events))

    def test_plot_coincidence_rates(self):
        # create result image
        self.result_image5_plot_coincidences_rates = plt.figure(
            "5.1", figsize=(20, 20))
        plot_coincidence_rates(
            self.spiketrains, self.UE, window_size=100 * pq.ms,
            window_step=10 * pq.ms, n_neurons=2,
            plot_params_user={'fsize': 14},
            plot_markers_user={'data_markercolor': ("m", "g")},
            position=(1, 1, 1))
        self.result_image5_plot_coincidences_rates.savefig(
            "/home/kramer/Documents/Studium Angewandte Mathematik und "
            "Informatik/INM6/PycharmProjects/viziphant-new/viziphant/tests/"
            "target_images/result_image5_plot_coincidence_rates.png")
        self.path_result_image5_plot_coincidence_rates = "target_images/" \
            "result_image5_plot_coincidence_rates.png"

        # assertion
        self.assertTrue(compare_images(
            self, self.path_target_image5_plot_coincidence_rates,
            self.path_result_image5_plot_coincidence_rates))

    def test_plot_statistical_significance(self):
        # create result image
        self.result_image6_plot_statistical_significance = plt.figure(
            "6.1", figsize=(20, 20))
        plot_statistical_significance(
            self.spiketrains, self.UE, ue.jointJ(0.05),
            window_size=100 * pq.ms, window_step=10 * pq.ms,
            n_neurons=2, plot_params_user={'fsize': 14},
            plot_markers_user={'data_markercolor': ("k", "b", "r")},
            position=(1, 1, 1))
        self.result_image6_plot_statistical_significance.savefig(
            "/home/kramer/Documents/Studium Angewandte Mathematik und "
            "Informatik/INM6/PycharmProjects/viziphant-new/viziphant/tests/"
            "target_images/result_image6_plot_statistical_significance.png")
        self.path_result_image6_plot_statistical_significance = "target_" \
            "images/result_image6_plot_statistical_significance.png"

        # assertion
        self.assertTrue(compare_images(
            self, self.path_target_image6_plot_statistical_significance,
            self.path_result_image6_plot_statistical_significance))

    def test_plot_unitary_events(self):
        # create result image
        self.result_image7_plot_unitary_events = plt.figure("7.1",
                                                            figsize=(20, 20))
        plot_unitary_events(
            self.spiketrains, self.UE, ue.jointJ(0.05), binsize=5 * pq.ms,
            window_size=100 * pq.ms, window_step=10 * pq.ms, n_neurons=2,
            plot_params_user={'fsize': 14}, plot_markers_user={},
            position=(1, 1, 1))
        self.result_image7_plot_unitary_events.savefig(
            "/home/kramer/Documents/Studium Angewandte Mathematik und "
            "Informatik/INM6/PycharmProjects/viziphant-new/viziphant/tests/"
            "target_images/result_image7_plot_unitary_events.png")
        self.path_result_image7_plot_unitary_events = "target_images/" \
            "result_image7_plot_unitary_events.png"

        # assertion
        self.assertTrue(compare_images(
            self, self.path_target_image7_plot_unitary_events,
            self.path_result_image7_plot_unitary_events))


if __name__ == '__main__':
    unittest.main()
