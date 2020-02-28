import io
import unittest

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from viziphant.spike_train_correlation import plot_corrcoef
from viziphant.tests.create_target.create_target_spike_train_correlation \
    import CORRCOEF_TARGET_PATH, get_default_corrcoef_matrix, \
    create_target_plot_correlation_coefficient


def image_difference(path_target_image, path_result_image):
    """
    Computes normalized images difference.

    Parameters
    ----------
    path_target_image : str or io.BytesIO
        The file-like target image.
    path_result_image : str or io.BytesIO
        The file-like result image to compare with the target.

    Returns
    -------
    diff_norm : float
        The L1-norm of the difference between two input images per pixel per
        channel.
    """
    # imread returns RGBA image
    target_image = mpimg.imread(path_target_image, format='png')
    result_image = mpimg.imread(path_result_image, format='png')
    diff_image = np.abs(target_image - result_image)
    diff_norm = np.linalg.norm(np.ravel(diff_image), ord=1)
    diff_norm = diff_norm / diff_image.size  # per pixel per channel
    return diff_norm


class SpikeTrainCorrelationTestCase(unittest.TestCase):
    def test_corroef(self):
        # TODO: remove creating target image function after setting up \
        #  git-lfs
        create_target_plot_correlation_coefficient()

        seaborn.set_style('ticks')
        result_image_corrcoef, axes2 = plt.subplots(
            1, 1, subplot_kw={'aspect': 'equal'})

        plot_corrcoef(get_default_corrcoef_matrix(),
                      axes2,
                      correlation_minimum=-1.,
                      correlation_maximum=1.,
                      colormap='bwr', color_bar_aspect=20,
                      color_bar_padding_fraction=.5)

        with io.BytesIO() as buf:
            result_image_corrcoef.savefig(buf, format="png")
            buf.seek(0)
            diff_norm = image_difference(str(CORRCOEF_TARGET_PATH), buf)
        tolerance = 1e-3
        self.assertLessEqual(diff_norm, tolerance)


if __name__ == '__main__':
    unittest.main()
