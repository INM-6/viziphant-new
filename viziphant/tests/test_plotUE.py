import tempfile
import unittest

import numpy as np
from viziphant.plotting_unitary_events_analysis import plot_UE
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


class UETestCase(unittest.TestCase):
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

    def test_plot_UE(self):
        # print("vorher")
        plot_UE(
            data=self.spiketrains, joint_suprise_dict=self.UE,
            joint_suprise_significance=ue.jointJ(0.05), binsize=5*pq.ms,
            window_size=100*pq.ms, window_step=10*pq.ms, n_neurons=2)
        # print("nachher")


if __name__ == '__main__':
    unittest.main()
