import matplotlib.pyplot as plt
import numpy as np

import elephant.unitary_event_analysis as ue


def plot_unitary_events_simplified(
        data, joint_surprise_dict, joint_surprise_significance, binsize,
        window_size, window_step, n_neurons):
    print("plotting Unitary Events (simplified)")
    # 1. set necessary variables
    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_window_position = ue._winpos(t_start, t_stop, window_size, window_step)
    n_trial = len(data)

    # 2. create a subplot in the already defined figure
    axes = plt.subplot(1, 1, 1)
    axes.set_title("Unitary Events simplified new")

    # 3.iterate over the data-set to ...
    for n in range(n_neurons):
        for trial, data_trial in enumerate(data):
            # 3.1 ... plot all spike events
            plt.plot(data_trial[n].rescale('ms').magnitude,
                     np.ones_like(data_trial[n].magnitude) * trial +
                     n * (n_trial + 1) + 1,
                     ls='None', marker='.', markersize=0.5, color="k")

            # 3.2 searching for unitary events locations in the
            # joint_surprise_dict (created by using jointJ_window_analysis()),
            # assuming that there the calculated surprise is greater/equal than
            # the surprise_significance (created by using jointJ())
            # former: sig_idx_win ;; -> significant_indices_in_analysis_window
            # now: indices_of_significant_JointSurprises
            indices_of_significant_JointSurprises = np.where(
                joint_surprise_dict['Js'] >= joint_surprise_significance)[0]

            # 3.3 if at least one index, representing a significant surprise,
            # exists, ...
            if len(indices_of_significant_JointSurprises) > 0:
                # 3.3.1 .. remove all duplicate elements in the
                # joint_surprise_dict['indices']['trial'] - array and sort them
                # to get ..
                # x: ???locations of possible significant correlation
                # between spike events??? or maybe
                # indices_of_unique_pattern_within_a_window_shape
                x = np.unique(
                    joint_surprise_dict['indices']['trial' + str(trial)])

                # 3.3.2 if at least one ...
                if len(x) > 0:
                    xx = []
                    for j in indices_of_significant_JointSurprises:
                        # 3.3.2.1 append from x the i-th element to xx, when
                        # i-th_ele*binsize is in the analysis-window, meaning
                        # >= t_winpos[j] AND < t_winpos[j] + window_size
                        # xx: indices_of_patter_in_analysis_window
                        xx = np.append(
                            xx,
                            x[np.where((x * binsize >= t_window_position[j]) &
                            (x * binsize < t_window_position[j] + window_size))
                            ])

                        # 3.3.2.2 plotting all unitary events with markers
                        plt.plot(np.unique(xx) * binsize,
                                 np.ones_like(np.unique(xx)) * trial +
                                 n * (n_trial + 1) + 1,
                                 ms=5, marker='s', ls='', markeredgecolor='r')

                        # print("indices_of_significant_JointSurprises: ",
                        #       indices_of_significant_JointSurprises, "\n",
                        #       "x: ", x, "\n",
                        #       "xx: ", xx, "\n")

        # horizontal separation line
        if n < n_neurons - 1:
            axes.axhline((trial + 2) * (n + 1))

        axes.set_xlim(
            (min(t_window_position) - window_size).rescale('ms').magnitude,
            (max(t_window_position) + window_size).rescale('ms').magnitude)
        axes.set_ylim(0, (trial + 2) * (n + 1) + 1)
    axes.set_xlabel('Time [ms]')
    axes.set_ylabel('Trial')
