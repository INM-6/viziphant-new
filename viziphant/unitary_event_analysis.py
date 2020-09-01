"""
Plotting function for unitary event analysis results.
"""
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

import elephant.unitary_event_analysis as ue

FigureUE = namedtuple(
    "AxesUE", "spike_events, spike_rates, coincident_events, "
              "coincidence_rates, statistical_significance, unitary_events")

params_dict_default = {
    # epochs to be marked on the time axis
    'events': {},
    # size of the figure
    'figsize': (10, 12),
    # id of the units
    'unit_ids': [1, 2],
    # horizontal white space between subplots
    'hspace': 1,
    # width white space between subplots
    'wspace': 0.5,
    # orientation (figure margins)
    'top': 0.9,
    'bottom': 0.1,
    'right': 0.9,
    'left': 0.1,
    # font size
    'fsize': 12,
    # the actual unit ids from the experimental recording
    'unit_real_ids': ['not specified', 'not specified'],
    # line width
    'lw': 2,
    # y limit for the surprise
    'S_ylim': (-3, 3),
    # size of the marker
    'marker_size': 5,
    # uniform time unit
    'time_unit': pq.ms,
    # uniform frequency unit
    'frequency_unit': 'Hz',
}


def plot_unitary_events(data, joint_surprise_dict, significance_level, binsize,
                        window_size, window_step, **plot_params_user):
    """
    Plots the results of unitary event analysis as a column of six subplots,
    comprised of raster plot, peri-stimulus time histogram, coincident event
    plot, coincidence rate plot, significance plot and unitary event plot,
    respectively.

    Parameters
    ----------
    data : list of list of neo.SpikeTrain
        A nested list of trails, neurons and there neo.SpikeTrain objects,
        respectively. This should be identical to the one used to generate
        joint_surprise_dict
    joint_surprise_dict : dict
        The output of elephant.unitary_event_analysis.jointJ_window_analysis
        function. The values of each key has the shape of
            different pattern hash --> 0-axis
            different window --> 1-axis
        Keys:
        -----
        Js : list of float
            JointSurprise of different given pattern within each window.
        indices : list of list of int
            A list of indices of pattern within each window.
        n_emp : list of int
        The empirical number of each observed pattern.
        n_exp : list of float
            The expected number of each pattern.
        rate_avg : list of float
            The average firing rate of each neuron.
    significance_level : float
        The significance threshold used to determine which coincident events
        are classified as unitary events within a window.
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
        figsize : tuple of int (default: (10, 12))
            The dimensions for the figure size.
        hspace : float (default: 1)
            The amount of height reserved for white space between subplots.
        wspace : float (default: 0.5)
            The amount of width reserved for white space between subplots.
        top : float (default: 0.9)
        bottom : float (default: 0.1)
        right : float (default: 0.9)
        left : float (default: 0.1)
            The sizes of the respective margin of the subplot in the figure.
        fsize : integer (default: 12)
            The size of the font
        unit_real_ids : list of integers (default: [1, 2])
            The unit ids form the experimental recording.
        lw: float (default: 2)
            The default line width.
        S_ylim : tuple of ints or floats (default: (-3, 3))
            The y-axis limits for the joint surprise plot.
        marker_size : integers (default: 5)
            The marker size for the coincidence and unitary events.
        time_unit : quantities (default: quantities.ms)
            The time unit used to rescale the spiketrains.
        frequency_unit : string (default: 'Hz')
            The frequency unit used to rescale the spikerates.
    Returns
    -------
    result : instance of namedtuple()
        The container for Axis objects generated by this function. Individual
        axes can be accessed using the respective identifiers:
        result.identifier
        Identifiers: spike_events_axes, spike_rates_axes,
                     coincidence_events_axes, coincidence_rates_axes,
                     statistical_significance_axes, unitary_events_axes
    """
    # update params_dict_default with user input
    params_dict = params_dict_default.copy()
    params_dict.update(plot_params_user)

    # rescale all spiketrains to the uniform time unit from params_dict
    for m in range(len(data)):
        for n in range(len(data[0])):
            data[m][n] = data[m][n].rescale(params_dict['time_unit'])

    # set common variables
    n_neurons = len(data[0])
    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, window_size, window_step)
    center_of_analysis_window = t_winpos + window_size / 2.
    n_trials = len(data)
    joint_surprise_significance = ue.jointJ(significance_level)
    xlim_left = (min(t_winpos)).magnitude
    xlim_right = (max(t_winpos) + window_size).magnitude

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
    y_ticks_list = [1]
    y_ticks_labels_list = [1]
    for n in range(n_neurons):
        y_ticks_list.append((n + 1) * n_trials + n)
        y_ticks_labels_list.append(n_trials)

    def mark_epochs(axes_name):
        """
        Marks epochs on the respective axis by creating a vertical line and
        shows the epoch's name under the last subplot. Epochs need to be
        defined in the plot_params_user dictionary.
        Parameters
        ----------
        axes_name : matplotlib.axes._subplots.AxesSubplot
            The axes in which the epochs will be marked.
        """
        for key in params_dict['events'].keys():
            for event_timepoint in params_dict['events'][key]:
                # check if epochs are between time-axis limits
                if ((xlim_left <= event_timepoint) and
                        (event_timepoint <= xlim_right)):
                    axes_name.axvline(event_timepoint, ls='-',
                                      lw=params_dict['lw'], color='r')
                    if axes_name.get_geometry()[2] == 6:
                        axes_name.text(x=event_timepoint, y=-54, s=key,
                                       fontsize=12, color='r',
                                       horizontalalignment='center')

    print('plotting Unitary Event Analysis ...')

    print('plotting Spike Events ...')
    axes1 = plt.subplot(6, 1, 1)
    axes1.set_title('Spike Events')
    for n in range(n_neurons):
        for trial, data_trial in enumerate(data):
            spike_events_on_timescale = data_trial[n].magnitude
            spike_events_on_trialscale = \
                np.full_like(data_trial[n].magnitude, trial) + \
                n * (n_trials + 1) + 1
            axes1.plot(spike_events_on_timescale, spike_events_on_trialscale,
                       ls='none', marker='.', color='k', markersize=0.5)
        if n < n_neurons - 1:
            axes1.axhline((trial + 2) * (n + 1), lw=0.5, color='k')
    axes1.set_xlim(xlim_left, xlim_right)
    axes1.set_ylim(0, (n_trials + 1) * n_neurons + 1)
    axes1.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes1.set_yticks(y_ticks_list)
    axes1.set_yticklabels(y_ticks_labels_list)
    axes1.text(xlim_right - 200, -34,
               f"Unit {params_dict['unit_real_ids'][0]}")
    axes1.text(xlim_right - 200, n_trials * n_neurons + 7,
               f"Unit {params_dict['unit_real_ids'][1]}")
    axes1.set_ylabel('Trial', fontsize=params_dict['fsize'])

    print('plotting Spike Rates ...')
    axes2 = plt.subplot(6, 1, 2, sharex=axes1)
    axes2.set_title('Spike Rates')
    # psth = peristimulus time histogram
    max_val_psth = 0
    for n in range(n_neurons):
        respective_rate_average = joint_surprise_dict['rate_avg'][:, n].\
            rescale(params_dict['frequency_unit'])
        axes2.plot(center_of_analysis_window, respective_rate_average,
                   label=f"Unit {params_dict['unit_real_ids'][n]}",
                   lw=params_dict['lw'])
        if max(joint_surprise_dict['rate_avg'][:, n]) > max_val_psth:
            max_val_psth = max(joint_surprise_dict['rate_avg'][:, n])
    axes2.set_xlim(xlim_left, xlim_right)
    max_val_psth = max_val_psth.rescale(
        params_dict['frequency_unit']).magnitude
    axes2.set_ylim(0, max_val_psth + max_val_psth/10)
    axes2.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes2.set_yticks([0, int(max_val_psth / 2), int(max_val_psth)])
    axes2.legend(fontsize=params_dict['fsize']//2)
    axes2.set_ylabel(f"({params_dict['frequency_unit']})",
                     fontsize=params_dict['fsize'])

    print('plotting Coincident Events ...')
    axes3 = plt.subplot(6, 1, 3, sharex=axes1)
    axes3.set_title('Coincident Events')
    for n in range(n_neurons):
        for trial, data_trial in enumerate(data):
            spike_events_on_timescale = data_trial[n].magnitude
            spike_events_on_trialscale = \
                np.full_like(data_trial[n].magnitude, trial) + \
                n * (n_trials + 1) + 1
            axes3.plot(spike_events_on_timescale, spike_events_on_trialscale,
                       ls='none', marker='.', color='k', markersize=0.5)
            indices_of_coincidence_events = \
                np.unique(joint_surprise_dict['indices']['trial' + str(trial)])
            coincidence_events_on_timescale = \
                indices_of_coincidence_events * binsize
            coincidence_events_on_trialscale = np.full_like(
                indices_of_coincidence_events, trial) + n * (n_trials + 1) + 1
            axes3.plot(coincidence_events_on_timescale,
                       coincidence_events_on_trialscale, ls='',
                       markersize=params_dict['marker_size'], marker='s',
                       markerfacecolor='none', markeredgecolor='c')
        if n < n_neurons - 1:
            axes3.axhline((trial + 2) * (n + 1), lw=0.5, color='k')
    axes3.set_xlim(xlim_left, xlim_right)
    axes3.set_ylim(0, (n_trials + 1) * n_neurons + 1)
    axes3.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes3.set_yticks(y_ticks_list)
    axes3.set_yticklabels(y_ticks_labels_list)
    axes3.set_ylabel('Trial', fontsize=params_dict['fsize'])

    print('plotting emp. and exp. Coincidence Rates ..')
    axes4 = plt.subplot(6, 1, 4, sharex=axes1)
    axes4.set_title('Coincidence Rates')
    empirical_coincidence_rate = joint_surprise_dict['n_emp'] / \
        (window_size.rescale('s').magnitude * n_trials)
    expected_coincidence_rate = joint_surprise_dict['n_exp'] / \
        (window_size.rescale('s').magnitude * n_trials)
    axes4.plot(center_of_analysis_window, empirical_coincidence_rate,
               label='Empirical', lw=params_dict['lw'], color='c')
    axes4.plot(center_of_analysis_window, expected_coincidence_rate,
               label='Expected', lw=params_dict['lw'], color='m')
    axes4.set_xlim(xlim_left, xlim_right)
    axes4.xaxis.set_major_locator(MaxNLocator(integer=True))
    y_ticks = axes4.get_ylim()
    axes4.set_yticks([y_ticks[0], y_ticks[1] / 2, y_ticks[1]])
    axes4.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes4.legend(fontsize=params_dict['fsize']//2)
    axes4.set_ylabel(f"({params_dict['frequency_unit']})",
                     fontsize=params_dict['fsize'])

    print('plotting Statistical Significance ...')
    axes5 = plt.subplot(6, 1, 5, sharex=axes1)
    axes5.set_title('Statistical Significance')
    joint_surprise_values = joint_surprise_dict['Js']
    axes5.plot(center_of_analysis_window, joint_surprise_values,
               lw=params_dict['lw'], color='k')
    axes5.set_xlim(xlim_left, xlim_right)
    axes5.set_ylim(params_dict['S_ylim'])
    axes5.axhline(joint_surprise_significance, ls='-', color='r')
    axes5.axhline(-joint_surprise_significance, ls='-', color='g')
    axes5.text(t_winpos[30], joint_surprise_significance + 0.3, '$\\alpha +$',
               color='r')
    axes5.text(t_winpos[30], -joint_surprise_significance - 0.9, '$\\alpha -$',
               color='g')
    axes5.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes5.set_yticks([ue.jointJ(1-significance_level), ue.jointJ(0.5),
                      ue.jointJ(significance_level)])
    axes5.set_yticklabels([1-significance_level, 0.5, significance_level])

    print('plotting Unitary Events ...')
    axes6 = plt.subplot(6, 1, 6, sharex=axes1)
    axes6.set_title('Unitary Events')
    for n in range(n_neurons):
        for trial, data_trial in enumerate(data):
            spike_events_on_timescale = data_trial[n].magnitude
            spike_events_on_trialscale = \
                np.full_like(data_trial[n].magnitude, trial) + \
                n * (n_trials + 1) + 1
            axes6.plot(spike_events_on_timescale, spike_events_on_trialscale,
                       ls='None', marker='.', markersize=0.5, color='k')
            indices_of_significant_joint_surprises = np.where(
                joint_surprise_dict['Js'] >= joint_surprise_significance)[0]
            if len(indices_of_significant_joint_surprises) > 0:
                indices_of_coincidence_events = np.unique(
                    joint_surprise_dict['indices']['trial' + str(trial)])
                if len(indices_of_coincidence_events) > 0:
                    indices_of_unitary_events = []
                    for j in indices_of_significant_joint_surprises:
                        coincidence_indices_greater_left_window_margin = \
                            indices_of_coincidence_events * binsize >= \
                            t_winpos[j]
                        coincidence_indices_smaller_right_window_margin = \
                            indices_of_coincidence_events * binsize < \
                            t_winpos[j] + window_size
                        coincidence_indices_in_actual_analysis_window = \
                            coincidence_indices_greater_left_window_margin & \
                            coincidence_indices_smaller_right_window_margin
                        indices_of_unitary_events = \
                            np.append(
                                indices_of_unitary_events,
                                indices_of_coincidence_events
                                [coincidence_indices_in_actual_analysis_window]
                                )
                    unitary_events_on_timescale = \
                        np.unique(indices_of_unitary_events) * binsize
                    unitary_events_on_trialscale = \
                        np.ones_like(np.unique(indices_of_unitary_events)) * \
                        trial + n * (n_trials + 1) + 1
                    axes6.plot(unitary_events_on_timescale,
                               unitary_events_on_trialscale,
                               markersize=params_dict['marker_size'],
                               marker='s', ls='', markerfacecolor='none',
                               markeredgecolor='r')
        if n < n_neurons - 1:
            axes6.axhline((trial + 2) * (n + 1), lw=0.5, color='k')
    axes6.set_xlim(xlim_left, xlim_right)
    axes6.set_ylim(0, (n_trials + 1) * n_neurons + 1)
    axes6.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes6.set_yticks(y_ticks_list)
    axes6.set_yticklabels(y_ticks_labels_list)
    axes6.set_ylabel('Trial', fontsize=params_dict['fsize'])

    # mark all epochs on all subplots and annotate all axes-subplots;;
    # add to all subplots x-axis label
    for n in range(6):
        axes_list = [axes1, axes2, axes3, axes4, axes5, axes6]
        letter_list = ['A', 'B', 'C', 'D', 'E', 'F']
        mark_epochs(eval(f"axes{n+1}"))
        axes = axes_list[n]
        letter = letter_list[n]
        axes.text(-0.05, 1.1, letter, transform=axes.transAxes,
                  size=params_dict['fsize'] + 5, weight='bold')
        axes.set_xlabel(f'Time ({params_dict["time_unit"].dimensionality})',
                         fontsize=params_dict['fsize'])

    result = FigureUE(axes1, axes2, axes3, axes4, axes5, axes6)
    return result
