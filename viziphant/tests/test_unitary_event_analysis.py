import io
import tempfile
import unittest
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq
import os

import elephant.unitary_event_analysis as ue
from viziphant.tests.create_target.target_unitary_event_analysis import \
    create_target_unitary_event_analysis
from viziphant.tests.utils.utils import TEST_DATA_DIR
from viziphant.tests.utils.utils import images_difference, check_integrity
from viziphant.unitary_event_analysis import plot_unitary_events

UE_DATASET_URL = "https://web.gin.g-node.org/INM-6/elephant-data/raw/master/" \
                 "dataset-1/dataset-1.h5"


class UETestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Download data
        filepath = TEST_DATA_DIR / Path(UE_DATASET_URL).name
        if not filepath.exists():
            urlretrieve(UE_DATASET_URL, filename=filepath)
        check_integrity(filepath, md5="0219a243be3b452cf0537dc0a333fb2e")

        # Load data and extract spiketrains
        block = neo.io.NeoHdf5IO(filepath).read_block()
        sts1 = block.segments[0].spiketrains
        sts2 = block.segments[1].spiketrains
        cls.spiketrains = np.vstack((sts1, sts2)).T
        cls.UE = ue.jointJ_window_analysis(
            cls.spiketrains, binsize=5 * pq.ms, winsize=100 * pq.ms,
            winstep=10 * pq.ms, pattern_hash=[3])

    def _do_plot_UE(self):
        plot_params_user = {'events': {'Vision': [1000] * pq.ms,
                                       'Action': [1500] * pq.ms}}
        plot_unitary_events(self.spiketrains, joint_surprise_dict=self.UE,
                            significance_level=0.05, binsize=5 * pq.ms,
                            window_size=100 * pq.ms, window_step=10 * pq.ms,
                            n_neurons=2, **plot_params_user)

    def test_plot_UE(self):
        # Create target ;; TODO: once the target is fix/final, remove this
        target_path = create_target_unitary_event_analysis()

        with io.BytesIO() as buf:
            self._do_plot_UE()
            plt.savefig(buf, format="png")
            plt.show()
            buf.seek(0)
            diff_norm = images_difference(target_path, buf)
        tolerance = 3e-2
        self.assertLessEqual(diff_norm, tolerance)


if __name__ == '__main__':
    unittest.main()
