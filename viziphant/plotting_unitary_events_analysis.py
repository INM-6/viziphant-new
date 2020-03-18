import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

import elephant.unitary_event_analysis as ue

params_dict_default = {
    # params
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
    'top': 0.9,
    'bottom': 0.05,
    'right': 0.95,
    'left': 0.1,
    # font size
    'fsize': 12,
    # the actual unit ids from the experimental recording
    'unit_real_ids': [1, 2],
    # line width
    'lw': 0.5,
    # y limit for the surprise
    'S_ylim': (-3, 3),
    # size of the marker
    'marker_size': 5,
    # major tick width on the time scale (x-axis)
    'major_tick_width_time': 200,
    # number n of minor ticks between major ones on the time scale (x-axis)
    'number_minor_ticks_time': 1,
    # size of the y-ticks intervall for rasterplots
    'y_tick_interval': 15,
    # show figure
    'showfig': True,
    # save figure
    'savefig': False,
    # path, file name and format for saving the figure
    'path_filname_format': 'figure.pdf'
}


def plot_UE(data, joint_suprise_dict, significance_level, binsize,
            window_size, window_step, **plot_params_user):

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
                        wspace=params_dict['wspace'])

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

    def set_xticks(axes_name):
        axes_name.xaxis.set_major_locator(
            MultipleLocator(params_dict['major_tick_width_time']))
        axes_name.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axes_name.xaxis.set_minor_locator(
            MultipleLocator(params_dict['major_tick_width_time'] /
                            (params_dict['number_minor_ticks_time'] + 1)))

    def mark_epochs(axes_name):
        for key in params_dict['events'].keys():
            for e_val in params_dict['events'][key]:
                axes_name.axvline(e_val, ls='-', lw=params_dict['lw'],
                                  color='r')
                if axes_name.get_geometry()[2] == 6:
                    axes_name.text(x=e_val-10, y=-65, s=key, fontsize=9,
                                   color='r')

    print('plotting Unitary Event Analysis ...')

    print('plotting Spike Events ...')
    axes1 = plt.subplot(6, 1, 1)
    axes1.set_title('Spike Events')
    for n in range(n_neurons):
        for trial, data_trial in enumerate(data):
            axes1.plot(data_trial[n].rescale('ms').magnitude,
                       np.ones_like(data_trial[n].magnitude) * trial +
                       n * (n_trail + 1) + 1, ls='none', marker='.',
                       color='k', markersize=0.5)
        if n < n_neurons - 1:
            axes1.axhline((trial + 2) * (n + 1), lw=params_dict['lw'],
                          color='b')
    axes1.set_xlim(xlim_left, xlim_right)
    axes1.set_ylim(0, (trial + 2) * (n + 1) + 1)
    set_xticks(axes1)
    axes1.set_yticks(y_ticks_list)
    axes1.set_yticklabels(y_ticks_labels_list, fontsize=params_dict['fsize'])
    for n in range(n_neurons):
        axes1.text(xlim_right + 20, n * (n_trail + 1),
                   f"Unit {params_dict['unit_real_ids'][n]}")
    axes1.set_ylabel('Trial', fontsize=params_dict['fsize'])

    print('plotting Spike Rates ...')
    axes2 = plt.subplot(6, 1, 2, sharex=axes1)
    axes2.set_title('Spike Rates')
    # psth = peristimulu time histogram
    max_val_psth = 0
    for n in range(n_neurons):
        axes2.plot(
            t_winpos + window_size / 2.,
            joint_suprise_dict['rate_avg'][:, n].rescale('Hz'),
            label=f"Unit {params_dict['unit_real_ids'][n]}",
            lw=params_dict['lw'])
        if max(joint_suprise_dict['rate_avg'][:, n]) > max_val_psth:
            max_val_psth = max(joint_suprise_dict['rate_avg'][:, n])
    axes2.set_xlim(xlim_left, xlim_right)
    max_val_psth = max_val_psth.rescale('Hz').magnitude
    axes2.set_ylim(0, max_val_psth + max_val_psth/10)
    set_xticks(axes2)
    axes2.set_yticks([0, int(max_val_psth / 2), int(max_val_psth)])
    mark_epochs(axes2)
    axes2.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    axes2.set_ylabel('(1/s)', fontsize=params_dict['fsize'])

    print('plotting Coincidence Events ...')
    axes3 = plt.subplot(6, 1, 3, sharex=axes1)
    axes3.set_title('Coincidence Events')
    for n in range(n_neurons):
        for trial, data_trial in enumerate(data):
            axes3.plot(
                data_trial[n].rescale('ms').magnitude,
                np.ones_like(data_trial[n].magnitude) * trial +
                n * (n_trail + 1) + 1, ls='none', marker='.', color='k',
                markersize=0.5)
            axes3.plot(
                np.unique(joint_suprise_dict['indices']['trial' + str(trial)])
                * binsize,
                np.ones_like(np.unique(joint_suprise_dict['indices'][
                    'trial' + str(trial)])) * trial + n * (n_trail + 1) + 1,
                ls='', markersize=params_dict['marker_size'],
                marker='s', markerfacecolor='none', markeredgecolor='c')
        if n < n_neurons - 1:
            axes3.axhline((trial + 2) * (n + 1), lw=params_dict['lw'],
                          color='b')
    axes3.set_xlim(xlim_left, xlim_right)
    axes3.set_ylim(0, (trial + 2) * (n + 1) + 1)
    set_xticks(axes3)
    axes3.set_yticks(y_ticks_list)
    axes3.set_yticklabels(y_ticks_labels_list, fontsize=params_dict['fsize'])
    mark_epochs(axes3)
    axes3.set_ylabel('Trial', fontsize=params_dict['fsize'])

    print('plotting Coincidence Rates ..')
    axes4 = plt.subplot(6, 1, 4, sharex=axes1)
    axes4.set_title('Coincidence Rates')
    axes4.plot(t_winpos + window_size / 2., joint_suprise_dict['n_emp'] /
               (window_size.rescale('s').magnitude * n_trail),
               label='empirical', lw=params_dict['lw'], color='c')
    axes4.plot(t_winpos + window_size / 2., joint_suprise_dict['n_exp'] /
               (window_size.rescale('s').magnitude * n_trail),
               label='expected', lw=params_dict['lw'], color='m')
    axes4.set_xlim(xlim_left, xlim_right)
    set_xticks(axes4)
    y_ticks = axes4.get_ylim()
    axes4.set_yticks([0, y_ticks[1] / 2, y_ticks[1]])
    mark_epochs(axes4)
    axes4.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    axes4.set_ylabel('(1/s)', fontsize=params_dict['fsize'])

    print('plotting Statistical Significance ...')
    axes5 = plt.subplot(6, 1, 5, sharex=axes1)
    axes5.set_title('Statistical Significance')
    axes5.plot(t_winpos + window_size / 2., joint_suprise_dict['Js'],
               lw=params_dict['lw'], color='k')
    axes5.set_xlim(xlim_left, xlim_right)
    axes5.set_ylim(params_dict['S_ylim'])
    axes5.axhline(joint_suprise_significance, ls='-', color='r')
    axes5.axhline(-joint_suprise_significance, ls='-', color='b')
    axes5.text(t_winpos[30], joint_suprise_significance + 0.3, '$\\alpha +$',
               color='r')
    axes5.text(t_winpos[30], -joint_suprise_significance - 0.9, '$\\alpha -$',
               color='b')
    set_xticks(axes5)
    axes5.set_yticks([ue.jointJ(0.99), ue.jointJ(0.5), ue.jointJ(0.01)])
    mark_epochs(axes5)
    axes5.set_yticklabels([0.99, 0.5, 0.01])

    print('plottting Unitary Events ...')
    axes6 = plt.subplot(6, 1, 6, sharex=axes1)
    axes6.set_title('Unitary Events')
    # sig_inx_win: indices of significant JointSurprises within a window
    # x: indices of coincidence events in analysis window of a trail
    # xx: indices_of_unitary_events_in_analysis_window_of_a_trial
    for n in range(n_neurons):
        for trial, data_trial in enumerate(data):
            axes6.plot(data_trial[n].rescale('ms').magnitude,
                       np.ones_like(data_trial[n].magnitude) * trial +
                       n * (n_trail + 1) + 1, ls='None', marker='.',
                       markersize=0.5, color='k')
            sig_idx_win = np.where(
                joint_suprise_dict['Js'] >= joint_suprise_significance)[0]
            if len(sig_idx_win) > 0:
                x = np.unique(
                    joint_suprise_dict['indices']['trial' + str(trial)])
                if len(x) > 0:
                    xx = []
                    for j in sig_idx_win:
                        xx = np.append(xx, x[(x * binsize >= t_winpos[j]) &
                            (x * binsize < t_winpos[j] + window_size)])
                    axes6.plot(np.unique(xx) * binsize, np.ones_like(
                               np.unique(xx)) * trial + n * (n_trail + 1) + 1,
                               markersize=params_dict['marker_size'],
                               marker='s', ls='', markerfacecolor='none',
                               markeredgecolor='r')
        if n < n_neurons - 1:
            axes6.axhline((trial + 2) * (n + 1), lw=params_dict['lw'],
                          color='b')
    axes6.set_xlim(xlim_left, xlim_right)
    axes6.set_ylim(0, (trial + 2) * (n + 1) + 1)
    set_xticks(axes6)
    axes6.set_yticks(y_ticks_list)
    axes6.set_yticklabels(y_ticks_labels_list, fontsize=params_dict['fsize'])
    mark_epochs(axes6)
    axes6.set_xlabel('Time [ms]', fontsize=params_dict['fsize'])
    axes6.set_ylabel('Trial', fontsize=params_dict['fsize'])

    if params_dict['savefig']:
        plt.savefig(params_dict['path_filename_format'])
        if not params_dict['showfig']:
            plt.cla()
            plt.close()
    if params_dict['showfig']:
        plt.show()
        plt.close()
