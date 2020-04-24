
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq

import elephant.unitary_event_analysis as ue
from viziphant.tests.utils.utils import TEST_DATA_DIR, TARGET_IMAGES_DIR
from viziphant.tests.utils.utils import check_integrity
from viziphant.unitary_event_analysis import \
    plot_unitary_events

UE_DATASET_URL = "https://web.gin.g-node.org/INM-6/elephant-data/raw/master/" \
                 "dataset-1/dataset-1.h5"
PLOT_UE_TARGET_PATH = TARGET_IMAGES_DIR / "target_plot_UE.png"

# Download data
filepath = TEST_DATA_DIR / Path(UE_DATASET_URL).name
if not filepath.exists():
    urlretrieve(UE_DATASET_URL, filename=filepath)
check_integrity(filepath, md5="0219a243be3b452cf0537dc0a333fb2e")

# Load data and extract spiketrains
block = neo.io.NeoHdf5IO(filepath).read_block()
sts1 = block.segments[0].spiketrains
sts2 = block.segments[1].spiketrains
spiketrains = np.vstack((sts1, sts2)).T
UE = ue.jointJ_window_analysis(
    spiketrains, binsize=5 * pq.ms, winsize=100 * pq.ms,
    winstep=10 * pq.ms, pattern_hash=[3])


def create_target_unitary_event_analysis():
    plot_params_user = {'events': {'Vision': [1000] * pq.ms,
                                   'Action': [1500] * pq.ms}}

    target = plot_unitary_events(
        spiketrains, joint_surprise_dict=UE, significance_level=0.05,
        binsize=5 * pq.ms, window_size=100 * pq.ms, window_step=10 * pq.ms,
        n_neurons=2, **plot_params_user)
    plt.savefig(PLOT_UE_TARGET_PATH)
    plt.show(target)


if __name__ == '__main__':
    create_target_unitary_event_analysis()
