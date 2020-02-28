import tempfile
import unittest

import numpy as np
from viziphant.plot_unitary_events_simplified import \
    plot_unitary_events_simplified
import viziphant.tests.create_targets.\
    create_target_unitary_events_simplified as create_target
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


class UnitaryEventAnalysisSimplifiedTestCase(unittest.TestCase):
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

        # create 8.figure with 1 subplot with plot_unitary_events_simplified():
        create_target.create_target_unitary_events_simplified()
        cls.path_target_image_plot_unitary_events_simplified = \
            create_target.PLOT_UNITARY_EVENTS_SIMPLIFIED_TARGET_PATH

    def test_plot_unitary_events_simplified(self):
        # create result image
        self.result_image_plot_unitary_events_simplified = plt.figure("1.1",
                                                           figsize=(20, 20))
        plot_unitary_events_simplified(
            self.spiketrains, self.UE, ue.jointJ(0.05), binsize=5 * pq.ms,
            window_size=100 * pq.ms, window_step=10 * pq.ms, n_neurons=2)
        self.path_result_image_plot_unitary_events_simplified = \
            tempfile.mkstemp(suffix=".png")[1]
        self.result_image_plot_unitary_events_simplified.savefig(
            self.path_result_image_plot_unitary_events_simplified)

        # assertion
        self.assertTrue(compare_images(
            self.path_target_image_plot_unitary_events_simplified,
            self.path_result_image_plot_unitary_events_simplified))


if __name__ == '__main__':
    unittest.main()
