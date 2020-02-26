import io
import unittest

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn

from viziphant.spike_train_correlation import plot_corrcoef
from viziphant.tests.create_target.create_target_spike_train_correlation \
    import CORRCOEF_TARGET_PATH, get_default_corrcoef_matrix, \
    create_target_plot_correlation_coefficient
from viziphant.tests.create_target.utils import check_integrity


def compare_images(path_target_image, path_result_image):
    target_image_as_array = mpimg.imread(path_target_image, format='png')
    result_image_as_array = mpimg.imread(path_result_image, format='png')
    the_same = (result_image_as_array == target_image_as_array).all()

    return the_same


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

        # TODO: alternative: get numpy pic directly from matplotlib figure
        with io.BytesIO() as buf:
            result_image_corrcoef.savefig(buf, format="png")
            buf.seek(0)
            check_integrity(CORRCOEF_TARGET_PATH,
                            md5="d41d8cd98f00b204e9800998ecf8427e")
            self.assertTrue(compare_images(CORRCOEF_TARGET_PATH, buf))


if __name__ == '__main__':
    unittest.main()
