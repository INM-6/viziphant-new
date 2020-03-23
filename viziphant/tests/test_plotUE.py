import os
import unittest

import matplotlib.image as mpimg
import neo
import numpy as np
import quantities as pq

import elephant.unitary_event_analysis as ue
from viziphant.plotting_unitary_events_analysis import plot_UE


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
        target_images_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "tests/target_images")
        PLOT_UE_TARGET_PATH = os.path.join(
            target_images_dir, "target_plot_ue.png")
        plot_UE(
            data=self.spiketrains, joint_surprise_dict=self.UE,
            significance_level=0.05, binsize=5*pq.ms,
            window_size=100*pq.ms, window_step=10*pq.ms, n_neurons=2,
            **{'events': {'Vision': [1000]*pq.ms, 'Action': [1500]*pq.ms},
               'savefig': True, 'showfig': False,
               'path_filename_format': PLOT_UE_TARGET_PATH})



if __name__ == '__main__':
    unittest.main()
