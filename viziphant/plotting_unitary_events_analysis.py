"""
Plotting function for unitary event analysis results.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               MaxNLocator)

import elephant.unitary_event_analysis as ue

params_dict_default = {
    # epochs to be marked on the time axis
    'events': {},
    # size of the figure
    'figsize': (12, 10),
    # id of the units
    'unit_ids': [0, 1],
    # horizontal white space between subplots
    'hspace': 1,
    # width white space between subplots
    'wspace': 0.5,
    # orientation (figure margins)
    'top': 0.95,
    'bottom': 0.1,
    'right': 0.87,
    'left': 0.08,
    # font size
    'fsize': 12,
    # the actual unit ids from the experimental recording
    'unit_real_ids': ['not specified', 'not specified'],
    # line width
    'lw': 0.5,
    # y limit for the surprise
    'S_ylim': (-3, 3),
    # size of the marker
    'marker_size': 5,
    # size of the y-ticks interval for rasterplots
    'y_tick_interval': 15,
    # show figure
    'showfig': True,
    # save figure
    'savefig': False,
    # path, file name and format for saving the figure
    'path_filname_format': 'figure.pdf'
}


def plot_UE(data, joint_surprise_dict, significance_level, binsize,
            window_size, window_step, **plot_params_user):
    """
    Plots the results of unitary event analysis as a column of six subplots,
    comprised of raster plot, peri-stimulus time histogram, coincident event
    plot, coincidence rate plot, significance plot and unitary event plot,
    respectively.

    Parameters
    ----------
    data : list of list of neo.SpikeTrain
        A nested list of trails, neurons and ther neo.SpikeTrain objects,
        respectively. This should be identical to the one used to generate
        joint_suprise_dict
    joint_surprise_dict : dict
        The output of elephant.unitary_event_analysis.jointJ_window_analysis
        function. The values of each key has the shape of
            different pattern hash --> 0-axis
            different window --> 1-axis
        Keys:
        -----
        Js : list of float
            JointSuprise of different given pattern within each window.
        indices : list of list of int
            A list of indices of patternwithin each window.
        n_emp : list of int
        The empirical number of each observed pattern.
        n_exp : list of float
            The expected number of each pattern.
        rate_avg : list of float
            The average firing rate of each neuron.
    significance_level : float
        The significance threshold used to determine which coincident evvents
        are classified ad unitary events wthin a window.
    binsize : quantities.Quantity
        The size of bins for discretizing spike trains. This value should be
        identical to the one used to generate joint_surprise_dict.
    window_size : quantities.Quantity
        The size of the analysis-window. This value should be identical to the
        one used to generate joint_surprise_dict.
    window_step : quantities.Quantity
        The size of the window step. This value should be identical to th one
        used to generate joint_surprise_dict.
    plot_params_user : dict
        A dictionary of plotting parameters used to update the default plotting
        parameter values.
        Keys:
        -----
        events : dictionary (default: {})
            Epochs to be marked on the time axis
            key: epochs name as string
            value: list of quantities.Quantity
        figsize : tuple of int (default: (12, 10))
            The dimensions for the figure size.
        hspace : float (default: 1)
            The amount of height reserved for white space between subplots.
        wspace : float (default: 0.5)
            The amaount of width reserved for white space between subplots.
        top : float (default: 0.95)
        bottom : float (default: 0.1)
        right : float (default: 0.87)
        left : float (default: 0.08)
            The sizes of the respective margin of the subplot in the figure.
        fsize : integer (default: 12)
            The size of the font
        unit_real_ids : list of integers (default: [1, 2])
            The unit ids form the experimental recording.
        lw: float (default: 0.5)
            The default line width.
        S_ylim : tuple of ints or floats (default: (-3, 3))
            The y-axis limits for the joint surprise plot.
        marker_size : integers (default: 5)
            The marker size for the coincidence and unitary events.
        y_tick_interval : integers (default: 15)
            The size of the interval between y-ticks (for rasterplots only)
        showfig : boolean (default: True)
            Displays the figure on screen if True.
        savefig : boolean (default: False)
            Saves the figure to disk if True.
        path_filname_format : string (default: figure.pdf)
            The path and the filename to save the figure. The format is
            inferred from the filename extension.
    """

    # # set common variables
    n_neurons = len(data[0])
    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, window_size, window_step)
    n_trail = len(data)
    joint_suprise_significance = ue.jointJ(significance_level)
    xlim_left = (min(t_winpos)).rescale('ms').magnitude
    xlim_right = (max(t_winpos) + window_size).rescale('ms').magnitude

    # update params_dict_default with user input
    params_dict = params_dict_default.copy()
    params_dict.update(plot_params_user)
    # TODO: maybe remove this checking user-entries part to a separate function
    if len(params_dict['unit_real_ids']) != n_neurons:
        raise ValueError(
            'length of unit_ids should be equal to number of neurons! \n'
            'Unit_Ids: ' + params_dict['unit_real_ids']
            + 'not equal number of neurons: ' + n_neurons)

    plt.figure(num=1, figsize=params_dict['figsize'])
    plt.subplots_adjust(hspace=params_dict['hspace'],
                        wspace=params_dict['wspace'],
                        top=params_dict['top'],
                        bottom=params_dict['bottom'],
                        left=params_dict['left'],
                        right=params_dict['right'])

    # set y-axis for raster plots with ticks and labels
    y_tick_interval = params_dict['y_tick_interval']
    y_ticks_list = []
    # find start y-tick position for each trail
    for yt1 in range(1, n_neurons * n_trail, n_trail + 1):
        y_ticks_list.append(yt1)
    # find in-between y-tick position for each trail with the specific interval
    for n in range(n_neurons):
        for yt2 in range(n * (n_trail + 1) + y_tick_interval,
                         (n + 1) * n_trail, y_tick_interval):
            y_ticks_list.append(yt2)
    y_ticks_list.sort()
    y_ticks_labels_list = [1]
    number_of_in_between_y_ticks_per_neuron = math.floor(
        n_trail / y_tick_interval)
    for i in range(number_of_in_between_y_ticks_per_neuron):
        y_ticks_labels_list.append((i + 1) * y_tick_interval)
    auxiliary_list = y_ticks_labels_list
    # adding n_neuron times the y_ticks_labels_list to itself, so that each
    # neuron has the same y_ticks_labels
    for i in range(n_neurons - 1):
        y_ticks_labels_list += auxiliary_list

    def mark_epochs(axes_name):
        """
        Marks epochs on the respective axis by creating a vertical line and
        showing the the epochs name under the last subplot. Epochs need to be
        defined in the plot_params_user dictionary.
        Parameters
        ----------
        axes_name : matplotlib.axes._subplots.AxesSubplot
            The axis on which the epochs will be marked.
        """
        for key in params_dict['events'].keys():
            for e_val in params_dict['events'][key]:
                axes_name.axvline(e_val, ls='-', lw=params_dict['lw'],
                                  color='r')
                if axes_name.get_geometry()[2] == 6:
                    axes_name.text(x=e_val-10*pq.ms, y=-65, s=key, fontsize=9,
                                   color='r')

    print('plotting Unitary Event Analysis ...')

    print('plotting Spike Events ...')
    axis1 = plt.subplot(6, 1, 1)
    axis1.set_title('Spike Events')
    for n in range(n_neurons):
        for trial, data_trial in enumerate(data):
            axis1.plot(data_trial[n].rescale('ms').magnitude,
                       np.ones_like(data_trial[n].magnitude) * trial +
                       n * (n_trail + 1) + 1, ls='none', marker='.',
                       color='k', markersize=0.5)
        if n < n_neurons - 1:
            axis1.axhline((trial + 2) * (n + 1), lw=params_dict['lw'],
                          color='b')
    axis1.set_xlim(xlim_left, xlim_right)
    axis1.set_ylim(0, (trial + 2) * (n + 1) + 1)
    axis1.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis1.set_yticks(y_ticks_list)
    axis1.set_yticklabels(y_ticks_labels_list, fontsize=params_dict['fsize'])
    for n in range(n_neurons):
        axis1.text(xlim_right + 20, n * (n_trail + 1),
                   f"Unit {params_dict['unit_real_ids'][n]}")
    axis1.set_ylabel('Trial', fontsize=params_dict['fsize'])

    print('plotting Spike Rates ...')
    axis2 = plt.subplot(6, 1, 2, sharex=axis1)
    axis2.set_title('Spike Rates')
    # psth = peristimulu time histogram
    max_val_psth = 0
    for n in range(n_neurons):
        axis2.plot(
            t_winpos + window_size / 2.,
            joint_surprise_dict['rate_avg'][:, n].rescale('Hz'),
            label=f"Unit {params_dict['unit_real_ids'][n]}",
            lw=params_dict['lw'])
        if max(joint_surprise_dict['rate_avg'][:, n]) > max_val_psth:
            max_val_psth = max(joint_surprise_dict['rate_avg'][:, n])
    axis2.set_xlim(xlim_left, xlim_right)
    max_val_psth = max_val_psth.rescale('Hz').magnitude
    axis2.set_ylim(0, max_val_psth + max_val_psth/10)
    axis2.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis2.set_yticks([0, int(max_val_psth / 2), int(max_val_psth)])
    mark_epochs(axis2)
    # TODO: rethink legend positions (also axis4);;beware of length of unit id
    axis2.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    axis2.set_ylabel('(1/s)', fontsize=params_dict['fsize'])

    print('plotting Coincidence Events ...')
    axis3 = plt.subplot(6, 1, 3, sharex=axis1)
    axis3.set_title('Coincidence Events')
    for n in range(n_neurons):
        for trial, data_trial in enumerate(data):
            axis3.plot(
                data_trial[n].rescale('ms').magnitude,
                np.ones_like(data_trial[n].magnitude) * trial +
                n * (n_trail + 1) + 1, ls='none', marker='.', color='k',
                markersize=0.5)
            axis3.plot(
                np.unique(joint_surprise_dict['indices']['trial' + str(trial)])
                * binsize,
                np.ones_like(np.unique(joint_surprise_dict['indices'][
                    'trial' + str(trial)])) * trial + n * (n_trail + 1) + 1,
                ls='', markersize=params_dict['marker_size'],
                marker='s', markerfacecolor='none', markeredgecolor='c')
        if n < n_neurons - 1:
            axis3.axhline((trial + 2) * (n + 1), lw=params_dict['lw'],
                          color='b')
    axis3.set_xlim(xlim_left, xlim_right)
    axis3.set_ylim(0, (trial + 2) * (n + 1) + 1)
    axis3.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis3.set_yticks(y_ticks_list)
    axis3.set_yticklabels(y_ticks_labels_list, fontsize=params_dict['fsize'])
    mark_epochs(axis3)
    axis3.set_ylabel('Trial', fontsize=params_dict['fsize'])

    print('plotting Coincidence Rates ..')
    axis4 = plt.subplot(6, 1, 4, sharex=axis1)
    axis4.set_title('Coincidence Rates')
    axis4.plot(t_winpos + window_size / 2., joint_surprise_dict['n_emp'] /
               (window_size.rescale('s').magnitude * n_trail),
               label='empirical', lw=params_dict['lw'], color='c')
    axis4.plot(t_winpos + window_size / 2., joint_surprise_dict['n_exp'] /
               (window_size.rescale('s').magnitude * n_trail),
               label='expected', lw=params_dict['lw'], color='m')
    axis4.set_xlim(xlim_left, xlim_right)
    axis4.xaxis.set_major_locator(MaxNLocator(integer=True))
    y_ticks = axis4.get_ylim()
    axis4.set_yticks([0, y_ticks[1] / 2, y_ticks[1]])
    mark_epochs(axis4)
    axis4.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    axis4.set_ylabel('(1/s)', fontsize=params_dict['fsize'])

    print('plotting Statistical Significance ...')
    axis5 = plt.subplot(6, 1, 5, sharex=axis1)
    axis5.set_title('Statistical Significance')
    axis5.plot(t_winpos + window_size / 2., joint_surprise_dict['Js'],
               lw=params_dict['lw'], color='k')
    axis5.set_xlim(xlim_left, xlim_right)
    axis5.set_ylim(params_dict['S_ylim'])
    axis5.axhline(joint_suprise_significance, ls='-', color='r')
    axis5.axhline(-joint_suprise_significance, ls='-', color='b')
    axis5.text(t_winpos[30], joint_suprise_significance + 0.3, '$\\alpha +$',
               color='r')
    axis5.text(t_winpos[30], -joint_suprise_significance - 0.9, '$\\alpha -$',
               color='b')
    axis5.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis5.set_yticks([ue.jointJ(0.99), ue.jointJ(0.5), ue.jointJ(0.01)])
    mark_epochs(axis5)
    axis5.set_yticklabels([0.99, 0.5, 0.01])

    print('plottting Unitary Events ...')
    axis6 = plt.subplot(6, 1, 6, sharex=axis1)
    axis6.set_title('Unitary Events')
    # sig_inx_win: indices of significant JointSurprises within a window
    # x: indices of coincidence events in analysis window of a trail
    # xx: indices_of_unitary_events_in_analysis_window_of_a_trial
    for n in range(n_neurons):
        for trial, data_trial in enumerate(data):
            axis6.plot(data_trial[n].rescale('ms').magnitude,
                       np.ones_like(data_trial[n].magnitude) * trial +
                       n * (n_trail + 1) + 1, ls='None', marker='.',
                       markersize=0.5, color='k')
            sig_idx_win = np.where(
                joint_surprise_dict['Js'] >= joint_suprise_significance)[0]
            if len(sig_idx_win) > 0:
                x = np.unique(
                    joint_surprise_dict['indices']['trial' + str(trial)])
                if len(x) > 0:
                    xx = []
                    for j in sig_idx_win:
                        xx = np.append(xx, x[(x * binsize >= t_winpos[j]) &
                            (x * binsize < t_winpos[j] + window_size)])
                    axis6.plot(np.unique(xx) * binsize, np.ones_like(
                               np.unique(xx)) * trial + n * (n_trail + 1) + 1,
                               markersize=params_dict['marker_size'],
                               marker='s', ls='', markerfacecolor='none',
                               markeredgecolor='r')
        if n < n_neurons - 1:
            axis6.axhline((trial + 2) * (n + 1), lw=params_dict['lw'],
                          color='b')
    axis6.set_xlim(xlim_left, xlim_right)
    axis6.set_ylim(0, (trial + 2) * (n + 1) + 1)
    axis6.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis6.set_yticks(y_ticks_list)
    axis6.set_yticklabels(y_ticks_labels_list, fontsize=params_dict['fsize'])
    mark_epochs(axis6)
    axis6.set_xlabel('Time [ms]', fontsize=params_dict['fsize'])
    axis6.set_ylabel('Trial', fontsize=params_dict['fsize'])

    if params_dict['savefig']:
        plt.savefig(params_dict['path_filename_format'])
        if not params_dict['showfig']:
            plt.cla()
            plt.close()
    if params_dict['showfig']:
        plt.show()
        plt.close()
