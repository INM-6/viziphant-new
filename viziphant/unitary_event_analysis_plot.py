import math
import numpy
import quantities as pq
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

import elephant.unitary_event_analysis as ue

# TODO: remember: *args, unpacking operator * returns a tuple not a list
#                 **kwargs, unpacking operator ** returns a dictionary
plot_params_and_markers_default = {
    # # params
    # epochs to be marked on the time axis
    'events': [],
    # id of the units
    'unit_ids': [0, 1],
    # horizontal white space between subplots
    'hspace': 1,
    # width white space between subplots
    'wspace': 0.5,
    # font size         #Schriftgroesse
    'fsize': 12,
    # the actual unit ids from the experimental recording
    'unit_real_ids': [1, 2],
    # line width
    'lw': 0.5,
    # y limit for the surprise
    'S_ylim': (-3, 3),
    # major tick width on the time scale
    'major_tick_width_time': 200,
    # number n of minor ticks between major ones on the time scale:
    'number_minor_ticks_time': 1,
    # boolean: weather vertical lines at the major ticks of the x-axis will be
    # added
    'boolean_vertical_lines': False,
    # color of the vertical lines at the major ticks of the x-axis
    'color_vertical_lines': "r",
    # # markers
    'data_symbol': ".",
    'data_markersize': 0.5,
    'data_markercolor': ("k", "b", "r"),
    'data_markerfacecolor': "none",
    'data_markeredgecolor': "none",
    'event_symbol': "s",
    'event_markersize': 5,
    'event_markercolor': "r",
    'event_markerfacecolor': "none",
    'event_markeredgecolor': "r",
}


def plot_unitary_event_full_analysis(
        data, joint_suprise_dict, joint_suprise_significance, binsize,
        window_size, window_step, n_neurons, position,
        **plot_params_and_markers_user):
    """
    Visualization of the results of the Unitary Event Analysis.

    Unitary Event (UE) analysis is a statistical method that
    enables to analyze in a time resolved manner excess spike correlation
    between simultaneously recorded neurons by comparing the empirical
    spike coincidences (precision of a few ms) to the expected number
    based on the firing rates of the neurons.
    The following plots will be created:
    - Spike Events (as rasterplot)
    - Spike Rates (as curve)
    - Coincident Events (as rasterplot with markers)
    - Empirical & Excpected Coincidences Rates (as curves)
    - Suprise or Statistical Significance (as curve with alpha-limits)
    - Unitary Events (as rasterplot with markers)

    Parameters
    ----------
    data: list of spiketrains
        list of spiketrains in different trials as representation of
        neural activity
    joint_suprise_dict: dictionary
        JointSuprise dictionary
    joint_suprise_significance: list of floats
        list of suprise measure
    binsize: Quantity scalar with the dimension time
       size of bins for descritizing spike trains
    window_size: Quantity scalar with dimension time
       size of the window of analysis
    window_step: Quantity scalar with dimension time
       size of the window step
    n_neurons: integer
        number of Neurons
    plot_params_and_markers_user: dictionary
        plotting parameters and marker properties from the user
    position: list of position tuples
        (posSpikeEvents(c,r,i), posSpikeRates(c,r,i),
         posCoincidenceEvents(c,r,i), posCoincidenceRates(c,r,i),
        posStatisticalSignificance(c,r,i), posUnitaryEvents(c,r,i))
        pos is a three integer-tuple, where the first integer is the number
        of rows, the second the number of columns, and the third the index
        of the subplot

    Returns
    -------
    NO


    Raises
    ------
        ???until now not investigated???

    Warns
    -----
        ???until now not investigated???

    Warning
    -------
        ???until now not investigated???

    See Also
    --------
        ???until now not investigated???



    References ### copy from elephant
    ----------
    [1] Gruen, Diesmann, Grammont, Riehle, Aertsen (1999) J Neurosci Methods,
        94(1): 67-79.
    [2] Gruen, Diesmann, Aertsen (2002a,b) Neural Comput, 14(1): 43-80; 81-19.
    [3] Gruen S, Riehle A, and Diesmann M (2003) Effect of cross-trial
        nonstationarity on joint-spike events
        Biological Cybernetics 88(5):335-351.
    [4] Gruen S (2009) Data-driven significance estimation of precise spike
        correlation. J Neurophysiology 101:1126-1140 (invited review)
    :copyright: Copyright 2015-2016 by the Elephant team,
     see `doc/authors.rst`.
    :license: Modified BSD, see LICENSE.txt for details.
    """

    # checking the user entries:
    """
    try:
        _checkingUserEntries_plot_UE(data, joint_suprise_dict, 
        joint_suprise_significance, binsize,
        window_size, window_step, n_neurons, plot_params_user, 
        plot_markers_user, position)
    except (TypeError, KeyError) as errors:
        print(errors)
        raise errors
    """

    # subplots format and marker properties
    plot_params_and_markers_dict = plot_params_and_markers_default.copy()
    plot_params_and_markers_dict.update(plot_params_and_markers_user)

    if len(plot_params_and_markers_dict['unit_real_ids']) != n_neurons:
        raise ValueError(
            'length of unit_ids should be equal to number of neurons! \n'
            'Unit_Ids: ' + plot_params_and_markers_dict['unit_real_ids']
            + 'ungleich NumOfNeurons: ' + n_neurons)

    if 'suptitle' in plot_params_and_markers_dict.keys():
        plt.suptitle("Trial aligned on " +
                     plot_params_and_markers_dict['suptitle'], fontsize=20)
    plt.subplots_adjust(hspace=plot_params_and_markers_dict['hspace'],
                        wspace=plot_params_and_markers_dict['wspace'])

    # default positions of the subplots
    default_positions = {
        'position_spike_events': (6, 1, 1),
        'position_spike_rates': (6, 1, 2),
        'position_coincidence_events': (6, 1, 3),
        'position_coincidence_rates': (6, 1, 4),
        'position_statistical_significance': (6, 1, 5),
        'position_unitary_events': (6, 1, 6)
    }


    # TODO: add checking length of position to checking- user entries -> Done
    # print("befor-for")
    # print(position, range(len(position)))
    key_list = default_positions.keys()
    for pos_user, key in zip(position, key_list):
        # print("default_positions[i]: (vor)", default_positions[i])
        default_positions[key] = pos_user
        # print("default_positions: (vor)", default_positions[i])
    # print("after-for")

    plot_spike_events(
        data, window_size, window_step, n_neurons, default_positions[
            'position_spike_events'], **plot_params_and_markers_dict)

    plot_spike_rates(
        data, joint_suprise_dict, window_size, window_step, n_neurons,
        default_positions['position_spike_rates'],
        **plot_params_and_markers_dict)

    plot_coincidence_events(
        data, joint_suprise_dict, binsize, window_size, window_step, n_neurons,
        default_positions['position_coincidence_events'],
        **plot_params_and_markers_dict)

    plot_coincidence_rates(
        data, joint_suprise_dict, window_size, window_step, n_neurons,
        default_positions['position_coincidence_rates'],
        **plot_params_and_markers_dict)

    plot_statistical_significance(
        data, joint_suprise_dict, joint_suprise_significance, window_size,
        window_step, n_neurons, default_positions[
            'position_statistical_significance'],
        **plot_params_and_markers_dict)

    plot_unitary_events(
        data, joint_suprise_dict, joint_suprise_significance, binsize,
        window_size, window_step, n_neurons, default_positions[
            'position_unitary_events'], **plot_params_and_markers_dict)

    plot_unitary_events_simplified(
        data, joint_suprise_dict, joint_suprise_significance, binsize,
        window_size, window_step, n_neurons)


def plot_spike_events(
        data, window_size, window_step, n_neurons, position,
        **plot_params_and_markers_user):
    """
    Visualization of the spike events of the Unitary Event Analysis.

    Spike events occure when neurons get triggered from their neighbour
    neurons, so that they start firering themselves. This firing can be
    measured with electrods.


    Parameters
    ----------
    data: list of spiketrains
        list of spiketrains in different trials as representation of
        neural activity
    window_size: Quantity scalar with dimension time
       size of the window of analysis
    window_step: Quantity scalar with dimension time
       size of the window step
    n_neurons: integer
        number of Neurons
    plot_params_and_markers_user: dictionary
        plotting parameters and marker properties from the user
    position: tuple
        position is a three integer-tuple, where the first integer is the
        number of rows, the second the number of columns, and the third the
        index of the subplot

    Returns
    -------
    NONE,
    but the following plot will be created:
    - Spike Events (as rasterplot)

    Raises
    ------
        ???until now not investigated???

    Warns
    -----
        ???until now not investigated???

    Warning
    -------
        ???until now not investigated???

    See Also
    --------
        ???until now not investigated???



    References ### copy from elephant
    ----------
    [1] Gruen, Diesmann, Grammont, Riehle, Aertsen (1999) J Neurosci Methods,
        94(1): 67-79.
    [2] Gruen, Diesmann, Aertsen (2002a,b) Neural Comput, 14(1): 43-80; 81-19.
    [3] Gruen S, Riehle A, and Diesmann M (2003) Effect of cross-trial
        nonstationarity on joint-spike events
        Biological Cybernetics 88(5):335-351.
    [4] Gruen S (2009) Data-driven significance estimation of precise spike
        correlation. J Neurophysiology 101:1126-1140 (invited review)
    :copyright: Copyright 2015-2016 by the Elephant team,
    see `doc/authors.rst`.
    :license: Modified BSD, see LICENSE.txt for details.
    """

    print('plotting Spike Events as raster plot')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, window_size, window_step)
    n_trail = len(data)

    # subplots format and marker properties
    plot_params_and_markers_dict = plot_params_and_markers_default.copy()
    plot_params_and_markers_dict.update(plot_params_and_markers_user)

    # TODO: get review for setting default_position values -> done for spike
    #  events, rest is left
    # default-values for row, column  and index of the subplot-position
    default_position = {'position_row': 1, 'position_column': 1,
                        'position_index_subplot': 1}
    key_list = default_position.keys()
    for pos_user, key in zip(position, key_list):
        default_position[key] = pos_user

    ax0 = plt.subplot(default_position['position_row'],
                      default_position['position_column'],
                      default_position['position_index_subplot'])

    ax0.set_title('Spike Events')
    for n in range(n_neurons):
        for trial, data_trail in enumerate(data):
            ax0.plot(data_trail[n].rescale('ms').magnitude,
                     numpy.ones_like(data_trail[n].magnitude) * trial + n * (
                n_trail + 1) + 1, ls='none',
                marker=plot_params_and_markers_dict['data_symbol'],
                color=plot_params_and_markers_dict['data_markercolor'][0],
                markersize=plot_params_and_markers_dict['data_markersize'])
        if n < n_neurons - 1:
            ax0.axhline((trial + 2) * (n + 1),
                        lw=plot_params_and_markers_dict['lw'], color='b')

    ax0.set_xlim((min(t_winpos) - window_size).rescale('ms').magnitude,
                 (max(t_winpos) + window_size).rescale('ms').magnitude)
    # TODO: better minimum for set_xlim -> Done? oder statt -window_size, 0
    ax0.set_ylim(0, (trial + 2) * (n + 1) + 1)

    if plot_params_and_markers_dict['boolean_vertical_lines']:
        x_line_vertical = MultipleLocator(
            plot_params_and_markers_dict['major_tick_width_time']).tick_values(
            t_start.magnitude, t_stop.magnitude)
        for xc in x_line_vertical:
            ax0.axvline(xc, lw=plot_params_and_markers_dict['lw'],
                        color=plot_params_and_markers_dict[
                            'color_vertical_lines'])

    ax0.xaxis.set_major_locator(
        MultipleLocator(plot_params_and_markers_dict['major_tick_width_time']))
    ax0.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax0.xaxis.set_minor_locator(
        MultipleLocator(plot_params_and_markers_dict['major_tick_width_time'] /
                        (plot_params_and_markers_dict
                         ['number_minor_ticks_time'] + 1)))

    # set y-axis
    y_ticks_list = []
    # finding y-tick position for trail no.1
    for yt1 in range(1, n_neurons * n_trail, n_trail + 1):
        y_ticks_list.append(yt1)
    # finding y-tick position for trail interval of 15
    for n in range(n_neurons):
        for yt2 in range(n * (n_trail + 1) + 15, (n + 1) * n_trail, 15):
            y_ticks_list.append(yt2)
    y_ticks_list.sort()

    y_ticks_labels_list = [1]
    # setting y-tick interval to 15 trails
    number_of_y_ticks_per_neuron = math.floor(n_trail / 15)
    for i in range(number_of_y_ticks_per_neuron):
        y_ticks_labels_list.append((i + 1) * 15)

    auxiliary_list = y_ticks_labels_list
    # adding n_neuron times the y_ticks_labels_list to itself, so that each
    # neuron has the same y_ticks_labels
    for i in range(n_neurons - 1):
        y_ticks_labels_list += auxiliary_list

    ax0.set_yticks(y_ticks_list)
    ax0.set_yticklabels(y_ticks_labels_list,
                        fontsize=plot_params_and_markers_dict['fsize'])

    x_lim = ax0.get_xlim()
    # # First version
    # ax0.text(x_lim[1], n_trail * 2 + 7, 'Neuron 2')
    # ax0.text(x_lim[1], -12, 'Neuron 1')
    # TODO: get review-critic for alternativ & add if it is good to
    #  coincidence_events and unitary_events
    # alternative for variable n_neuron (>2):
    for n in range(n_neurons):
        n_th_neuron = 'Neuron ' + str(n+1)
        ax0.text(x_lim[1] + 20, n * (n_trail + 1), n_th_neuron)

    ax0.set_xlabel('Time [ms]', fontsize=plot_params_and_markers_dict['fsize'])
    ax0.set_ylabel('Trial', fontsize=plot_params_and_markers_dict['fsize'])


def plot_spike_rates(
        data, joint_suprise_dict, window_size, window_step, n_neurons,
        position, **plot_params_and_markers_user):
    """
    Visualization of the spike rates of the Unitary Event Analysis.

    The spike rates represent the number of spikes in a defined time interval.

    Parameters
    ----------
    data: list
        list of spiketrains in different trials as representation of
        neural activity
    joint_suprise_dict: dictionary
        JointSuprise dictionary
    window_size: Quantity scalar with dimension time
       size of the window of analysis
    window_step: Quantity scalar with dimension time
       size of the window step
    n_neurons: integer
        number of Neurons
    plot_params_and_markers_user: dictionary
        plotting parameters and marker properties from the user
    position: tuple
        pos is a three integer-tuple, where the first integer is the number
         of rows, the second the number of columns, and the third the index
         of the subplot

    Returns
    -------
    NONE,
    but the following plot will be created:
    - Spike Rates (as curve)

    Raises
    ------
        ???until now not investigated???

    Warns
    -----
        ???until now not investigated???

    Warning
    -------
        ???until now not investigated???

    See Also
    --------
        ???until now not investigated???



    References ### copy from elephant
    ----------
    [1] Gruen, Diesmann, Grammont, Riehle, Aertsen (1999) J Neurosci Methods,
        94(1): 67-79.
    [2] Gruen, Diesmann, Aertsen (2002a,b) Neural Comput, 14(1): 43-80; 81-19.
    [3] Gruen S, Riehle A, and Diesmann M (2003) Effect of cross-trial
        nonstationarity on joint-spike events
        Biological Cybernetics 88(5):335-351.
    [4] Gruen S (2009) Data-driven significance estimation of precise spike
        correlation. J Neurophysiology 101:1126-1140 (invited review)
    :copyright: Copyright 2015-2016 by the Elephant team,
    see `doc/authors.rst`.
    :license: Modified BSD, see LICENSE.txt for details.
    """

    print('plotting Spike Rates as line plots')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, window_size, window_step)

    # subplots format and marker properties
    plot_params_and_markers_dict = plot_params_and_markers_default.copy()
    plot_params_and_markers_dict.update(plot_params_and_markers_user)

    ax1 = plt.subplot(position[0], position[1], position[2])
    ax1.set_title('Spike Rates')
    # initialize max_val_psth, psth = peristimulu time histogram
    # TODO: make max_val_psth variable -> Done ?
    max_val_psth = 0
    for n in range(n_neurons):
        ax1.plot(t_winpos + window_size / 2.,
                 joint_suprise_dict['rate_avg'][:, n].rescale('Hz'),
                 label='Neuron ' + str(plot_params_and_markers_dict
                                       ['unit_real_ids'][n]),
                 color=plot_params_and_markers_dict['data_markercolor'][n],
                 lw=plot_params_and_markers_dict['lw'])

        # print("max_val_psth (searching): ", max_val_psth)
        if max(joint_suprise_dict['rate_avg'][:, n]) > \
                max_val_psth:
            max_val_psth = max(joint_suprise_dict['rate_avg'][:, n])

    ax1.set_xlim((min(t_winpos) - window_size).rescale('ms').magnitude,
                 (max(t_winpos) + window_size).rescale('ms').magnitude)

    max_val_psth = max_val_psth.rescale('Hz').magnitude
    ax1.set_ylim(0, max_val_psth)

    if plot_params_and_markers_dict['boolean_vertical_lines']:
        x_line_vertical = MultipleLocator(
            plot_params_and_markers_dict['major_tick_width_time']).tick_values(
            t_start.magnitude, t_stop.magnitude)
        for xc in x_line_vertical:
            ax1.axvline(xc, lw=plot_params_and_markers_dict['lw'],
                        color=plot_params_and_markers_dict
                        ['color_vertical_lines'])

    ax1.xaxis.set_major_locator(
        MultipleLocator(plot_params_and_markers_dict['major_tick_width_time']))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax1.xaxis.set_minor_locator(
        MultipleLocator(plot_params_and_markers_dict['major_tick_width_time'] /
                        (plot_params_and_markers_dict
                         ['number_minor_ticks_time'] + 1)))
    ax1.set_yticks([0, int(max_val_psth / 2), int(max_val_psth)])

    ax1.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    ax1.set_xlabel('Time [ms]', fontsize=plot_params_and_markers_dict['fsize'])
    ax1.set_ylabel('(1/s)', fontsize=plot_params_and_markers_dict['fsize'])


def plot_coincidence_events(
        data, joint_suprise_dict, binsize, window_size, window_step, n_neurons,
        position, **plot_params_and_markers_user):
    """
    Visualization of the coincidence events of the Unitary Event Analysis.

    Coincidence events occur, when the observed neurons fire almost
    simultaneously, with a high probability of correlation.

    Parameters
    ----------
    data: list of spiketrains
        list of spiketrains in different trials as representation of
        neural activity
    joint_suprise_dict: dictionary
        JointSuprise dictionary
    binsize: Quantity scalar with the dimension time
       size of bins for descritizing spike trains
    window_size: Quantity scalar with dimension time
       size of the window of analysis
    window_step: Quantity scalar with dimension time
       size of the window step
    n_neurons: integer
        number of Neurons
    plot_params_and_markers_user: dictionary
        plotting parameters and marker properties from the user
    position: tuple
        pos is a three integer-tuple, where the first integer is the number
        of rows, the second the number of columns, and the third the index
        of the subplot

    Returns
    -------
    NONE,
    but the following plot will be created:
    - Coincident Events (as rasterplot with markers)

    Raises
    ------
        ???until now not investigated???

    Warns
    -----
        ???until now not investigated???

    Warning
    -------
        ???until now not investigated???

    See Also
    --------
        ???until now not investigated???



    References ### copy from elephant
    ----------
    [1] Gruen, Diesmann, Grammont, Riehle, Aertsen (1999) J Neurosci Methods,
        94(1): 67-79.
    [2] Gruen, Diesmann, Aertsen (2002a,b) Neural Comput, 14(1): 43-80; 81-19.
    [3] Gruen S, Riehle A, and Diesmann M (2003) Effect of cross-trial
        nonstationarity on joint-spike events
        Biological Cybernetics 88(5):335-351.
    [4] Gruen S (2009) Data-driven significance estimation of precise spike
        correlation. J Neurophysiology 101:1126-1140 (invited review)
    :copyright: Copyright 2015-2016 by the Elephant team,
    see `doc/authors.rst`.
    :license: Modified BSD, see LICENSE.txt for details.
    """

    print('plotting Raw Coincidences as raster plot '
          'with markers indicating the Coincidences')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, window_size, window_step)
    n_trail = len(data)

    # subplots format and marker properties
    plot_params_and_markers_dict = plot_params_and_markers_default.copy()
    plot_params_and_markers_dict.update(plot_params_and_markers_user)

    ax2 = plt.subplot(position[0], position[1], position[2])
    ax2.set_title('Coincidence Events')
    for n in range(n_neurons):
        for tr, data_tr in enumerate(data):
            ax2.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) *
                     tr + n * (n_trail + 1) + 1, ls='None',
                     marker=plot_params_and_markers_dict['data_symbol'],
                     markersize=plot_params_and_markers_dict[
                         'data_markersize'], color=
                     plot_params_and_markers_dict['data_markercolor'][0])
            ax2.plot(numpy.unique(
                joint_suprise_dict['indices']['trial' + str(tr)]) * binsize,
                numpy.ones_like(numpy.unique(
                    joint_suprise_dict['indices']['trial' + str(tr)]))
                * tr + n * (n_trail + 1) + 1,
                ls='', ms=plot_params_and_markers_dict['event_markersize'],
                marker=plot_params_and_markers_dict['event_symbol'],
                markerfacecolor=plot_params_and_markers_dict
                ['event_markerfacecolor'],
                markeredgecolor=plot_params_and_markers_dict
                ['event_markeredgecolor'])
        if n < n_neurons - 1:
            ax2.axhline((tr + 2) * (n + 1),
                        lw=plot_params_and_markers_dict['lw'], color='b')
    ax2.set_xlim((min(t_winpos) - window_size).rescale('ms').magnitude,
                 (max(t_winpos) + window_size).rescale('ms').magnitude)
    ax2.set_ylim(0, (tr + 2) * (n + 1) + 1)

    if plot_params_and_markers_dict['boolean_vertical_lines']:
        x_line_veritcal = MultipleLocator(
            plot_params_and_markers_dict['major_tick_width_time']).tick_values(
            t_start.magnitude, t_stop.magnitude)
        for xc in x_line_veritcal:
            ax2.axvline(xc, lw=plot_params_and_markers_dict['lw'],
                        color=plot_params_and_markers_dict
                        ['color_vertical_lines'])

    ax2.xaxis.set_major_locator(
        MultipleLocator(plot_params_and_markers_dict['major_tick_width_time']))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax2.xaxis.set_minor_locator(
        MultipleLocator(plot_params_and_markers_dict['major_tick_width_time'] /
                        (plot_params_and_markers_dict
                         ['number_minor_ticks_time'] + 1)))
    # set y-axis
    y_ticks_list = []
    # finding y-tick position for trail no.1
    for yt1 in range(1, n_neurons * n_trail, n_trail + 1):
        y_ticks_list.append(yt1)
    # finding y-tick position for trail interval of 15
    for n in range(n_neurons):
        for yt2 in range(n * (n_trail + 1) + 15, (n + 1) * n_trail, 15):
            y_ticks_list.append(yt2)
    y_ticks_list.sort()

    y_ticks_labels_list = [1]
    # setting y-tick interval to 15 trails
    number_of_y_ticks_per_neuron = math.floor(n_trail / 15)
    for i in range(number_of_y_ticks_per_neuron):
        y_ticks_labels_list.append((i + 1) * 15)

    auxiliary_list = y_ticks_labels_list
    # adding n_neuron times the y_ticks_labels_list to itself, so that each
    # neuron has the same y_ticks_labels
    for i in range(n_neurons - 1):
        y_ticks_labels_list += auxiliary_list

    ax2.set_yticks(y_ticks_list)
    ax2.set_yticklabels(y_ticks_labels_list,
                        fontsize=plot_params_and_markers_dict['fsize'])

    ax2.set_xlabel('Time [ms]', fontsize=plot_params_and_markers_dict['fsize'])
    ax2.set_ylabel('Trial', fontsize=plot_params_and_markers_dict['fsize'])


def plot_coincidence_rates(
        data, joint_suprise_dict, window_size, window_step, n_neurons,
        position, **plot_params_and_markers_user):
    """
    Visualization of the coincidence rates of the Unitary Event Analysis.

    There are two different rates:
        -emirical: represent the experimental measure results
        of the coincidence rates
        -expected: represent the theoretical calculus results
        of the coincidence rates

    Parameters
    ----------
    data: list of spiketrains
        list of spiketrains in different trials as representation of
        neural activity
    joint_suprise_dict: dictionary
        JointSuprise dictionary
    window_size: Quantity scalar with dimension time
       size of the window of analysis
    window_step: Quantity scalar with dimension time
       size of the window step
    n_neurons: integer
        number of Neurons
    plot_params_and_markers_user: dictionary
        plotting parameters and marker properties from the user
    position: tuple
        pos is a three integer-tuple, where the first integer is the number
        of rows, the second the number of columns, and the third the index
        of the subplot

    Returns
    -------
    NONE,
    but the following plot will be created:
    - Empirical & Excpected Coincidences Rates (as curves)

    Raises
    ------
        ???until now not investigated???

    Warns
    -----
        ???until now not investigated???

    Warning
    -------
        ???until now not investigated???

    See Also
    --------
        ???until now not investigated???



    References ### copy from elephant
    ----------
    [1] Gruen, Diesmann, Grammont, Riehle, Aertsen (1999) J Neurosci Methods,
        94(1): 67-79.
    [2] Gruen, Diesmann, Aertsen (2002a,b) Neural Comput, 14(1): 43-80; 81-19.
    [3] Gruen S, Riehle A, and Diesmann M (2003) Effect of cross-trial
        nonstationarity on joint-spike events
        Biological Cybernetics 88(5):335-351.
    [4] Gruen S (2009) Data-driven significance estimation of precise spike
        correlation. J Neurophysiology 101:1126-1140 (invited review)
    :copyright: Copyright 2015-2016 by the Elephant team,
    see `doc/authors.rst`.
    :license: Modified BSD, see LICENSE.txt for details.
    """

    print('plotting empirical and expected coincidences rate as line plots')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, window_size, window_step)
    n_trail = len(data)

    # subplots format and marker properties
    plot_params_and_markers_dict = plot_params_and_markers_default.copy()
    plot_params_and_markers_dict.update(plot_params_and_markers_user)

    if len(plot_params_and_markers_dict['unit_real_ids']) != n_neurons:
        raise ValueError(
            'length of unit_ids should be equal to number of neurons! \n'
            'Unit_Ids: ' + plot_params_and_markers_dict[
                'unit_real_ids'] + 'ungleich NumOfNeurons: ' + n_neurons)

    ax3 = plt.subplot(position[0], position[1], position[2])
    ax3.set_title('Coincidence Rates')
    ax3.plot(t_winpos + window_size / 2.,
             joint_suprise_dict['n_emp'] / (
                 window_size.rescale('s').magnitude * n_trail),
             label='empirical', lw=plot_params_and_markers_dict['lw'],
             color=plot_params_and_markers_dict['data_markercolor'][0])
    ax3.plot(t_winpos + window_size / 2.,
             joint_suprise_dict['n_exp'] / (
                 window_size.rescale('s').magnitude * n_trail),
             label='expected', lw=plot_params_and_markers_dict['lw'],
             color=plot_params_and_markers_dict['data_markercolor'][1])
    ax3.set_xlim((min(t_winpos) - window_size).rescale('ms').magnitude,
                 (max(t_winpos) + window_size).rescale('ms').magnitude)

    if plot_params_and_markers_dict['boolean_vertical_lines']:
        x_line_vertical = MultipleLocator(
            plot_params_and_markers_dict['major_tick_width_time']).tick_values(
            t_start.magnitude, t_stop.magnitude)
        for xc in x_line_vertical:
            ax3.axvline(xc, lw=plot_params_and_markers_dict['lw'],
                        color=plot_params_and_markers_dict
                        ['color_vertical_lines'])

    ax3.xaxis.set_major_locator(
        MultipleLocator(plot_params_and_markers_dict['major_tick_width_time']))
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax3.xaxis.set_minor_locator(
        MultipleLocator(plot_params_and_markers_dict['major_tick_width_time'] /
                        (plot_params_and_markers_dict
                         ['number_minor_ticks_time'] + 1)))
    y_ticks = ax3.get_ylim()
    ax3.set_yticks([0, y_ticks[1] / 2, y_ticks[1]])

    ax3.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    ax3.set_xlabel('Time [ms]', fontsize=plot_params_and_markers_dict['fsize'])
    ax3.set_ylabel('(1/s)', fontsize=plot_params_and_markers_dict['fsize'])


def plot_statistical_significance(
        data, joint_suprise_dict, joint_suprise_significance, window_size,
        window_step, n_neurons, position, **plot_params_and_markers_user):

    """
    Visualization of the statistical significance
    of the Unitary Event Analysis.



    Parameters
    ----------
    data: list of spiketrains
        list of spiketrains in different trials as representation of
        neural activity
    joint_suprise_dict: dictionary
        JointSuprise dictionary
    joint_suprise_significance: list of floats
        list of suprise measure
    window_size: Quantity scalar with dimension time
       size of the window of analysis
    window_step: Quantity scalar with dimension time
       size of the window step
    n_neurons: integer
        number of Neurons
    plot_params_and_markers_user: dictionary
        plotting parameters and marker properties from the user
    position: tuple
        pos is a three integer-tuple, where the first integer is the number
        of rows, the second the number of columns, and the third the index
        of the subplot

    Returns
    -------
    NONE,
    but the following plot will be created:
    - Suprise or Statistical Significance (as curve with alpha-limits)

    Raises
    ------
        ???until now not investigated???

    Warns
    -----
        ???until now not investigated???

    Warning
    -------
        ???until now not investigated???

    See Also
    --------
        ???until now not investigated???



    References ### copy from elephant
    ----------
    [1] Gruen, Diesmann, Grammont, Riehle, Aertsen (1999) J Neurosci Methods,
        94(1): 67-79.
    [2] Gruen, Diesmann, Aertsen (2002a,b) Neural Comput, 14(1): 43-80; 81-19.
    [3] Gruen S, Riehle A, and Diesmann M (2003) Effect of cross-trial
        nonstationarity on joint-spike events
        Biological Cybernetics 88(5):335-351.
    [4] Gruen S (2009) Data-driven significance estimation of precise spike
        correlation. J Neurophysiology 101:1126-1140 (invited review)
    :copyright: Copyright 2015-2016 by the Elephant team,
    see `doc/authors.rst`.
    :license: Modified BSD, see LICENSE.txt for details.
    """

    print('plotting Surprise/Statistical Significance as line plot')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, window_size, window_step)

    alpha = 0.5

    # subplots format and marker properties
    plot_params_and_markers_dict = plot_params_and_markers_default.copy()
    plot_params_and_markers_dict.update(plot_params_and_markers_user)

    if len(plot_params_and_markers_dict['unit_real_ids']) != n_neurons:
        raise ValueError('length of unit_ids should be equal to '
                         'number of neurons! \nUnit_Ids: ' +
                         plot_params_and_markers_dict['unit_real_ids'] +
                         'ungleich NumOfNeurons: ' + n_neurons)

    ax4 = plt.subplot(position[0], position[1], position[2])
    ax4.set_title('Statistical Significance')
    ax4.plot(t_winpos + window_size / 2., joint_suprise_dict['Js'],
             lw=plot_params_and_markers_dict['lw'],
             color=plot_params_and_markers_dict['data_markercolor'][0])
    ax4.set_xlim((min(t_winpos) - window_size).rescale('ms').magnitude,
                 (max(t_winpos) + window_size).rescale('ms').magnitude)
    ax4.set_ylim(plot_params_and_markers_dict['S_ylim'])

    ax4.axhline(joint_suprise_significance, ls='-',
                color=plot_params_and_markers_dict['data_markercolor'][1])
    ax4.axhline(-joint_suprise_significance, ls='-',
                color=plot_params_and_markers_dict['data_markercolor'][2])
    ax4.text(t_winpos[30], joint_suprise_significance + 0.3, '$\\alpha +$',
             color=plot_params_and_markers_dict['data_markercolor'][1])
    ax4.text(t_winpos[30], -joint_suprise_significance - 0.5, '$\\alpha -$',
             color=plot_params_and_markers_dict['data_markercolor'][2])

    if plot_params_and_markers_dict['boolean_vertical_lines']:
        x_line_vertical = MultipleLocator(
            plot_params_and_markers_dict['major_tick_width_time']).tick_values(
            t_start.magnitude, t_stop.magnitude)
        for xc in x_line_vertical:
            ax4.axvline(xc, lw=plot_params_and_markers_dict['lw'],
                        color=plot_params_and_markers_dict
                        ['color_vertical_lines'])

    ax4.xaxis.set_major_locator(
        MultipleLocator(plot_params_and_markers_dict['major_tick_width_time']))
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax4.xaxis.set_minor_locator(
        MultipleLocator(plot_params_and_markers_dict['major_tick_width_time'] /
                        (plot_params_and_markers_dict
                         ['number_minor_ticks_time'] + 1)))
    ax4.set_yticks([ue.jointJ(0.99), ue.jointJ(0.5), ue.jointJ(0.01)])

    ax4.set_xlabel('Time [ms]', fontsize=plot_params_and_markers_dict['fsize'])
    # TODO: how to set yticklabels correct: bottem = 0, top=1 ?
    # TODO: should alpha be a variable or like now constant 0.5 ?
    ax4.set_yticklabels([alpha - 0.5, alpha, alpha + 0.5])


def plot_unitary_events(
        data, joint_suprise_dict, joint_suprise_significance, binsize,
        window_size, window_step, n_neurons, position,
        **plot_params_and_markers_user):

    """
    Visualization of the unitary events of the Unitary Event Analysis.

    Unitary events are coincidence events that are not expected.

    Parameters
    ----------
    data: list of spiketrains
        list of spiketrains in different trials as representation of
        neural activity
    joint_suprise_dict: dictionary
        JointSuprise dictionary
    joint_suprise_significance: list of floats
        list of suprise measure
    binsize: Quantity scalar with the dimension time
       size of bins for descritizing spike trains
    window_size: Quantity scalar with dimension time
       size of the window of analysis
    window_step: Quantity scalar with dimension time
       size of the window step
    n_neurons: integer
        number of Neurons
    plot_params_and_markers_user: dictionary
        plotting parameters and marker properties from the user
    position: tuple
        pos is a three integer-tuple, where the first integer is the number
        of rows, the second the number of columns, and the third the index
        of the subplot

    Returns
    -------
    NONE,
    but the following plots will be created:
    - Unitary Events (as rasterplot with markers)

    Raises
    ------
        ???until now not investigated???

    Warns
    -----
        ???until now not investigated???

    Warning
    -------
        ???until now not investigated???

    See Also
    --------
        ???until now not investigated???



    References ### copy from elephant
    ----------
    [1] Gruen, Diesmann, Grammont, Riehle, Aertsen (1999) J Neurosci Methods,
        94(1): 67-79.
    [2] Gruen, Diesmann, Aertsen (2002a,b) Neural Comput, 14(1): 43-80; 81-19.
    [3] Gruen S, Riehle A, and Diesmann M (2003) Effect of cross-trial
        nonstationarity on joint-spike events
        Biological Cybernetics 88(5):335-351.
    [4] Gruen S (2009) Data-driven significance estimation of precise spike
        correlation. J Neurophysiology 101:1126-1140 (invited review)
    :copyright: Copyright 2015-2016 by the Elephant team,
    see `doc/authors.rst`.
    :license: Modified BSD, see LICENSE.txt for details.
    """

    print('plotting Unitary Events as raster plot '
          'with markers indicating the Unitary Events')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, window_size, window_step)
    n_trail = len(data)

    # subplots format and marker properties
    plot_params_and_markers_dict = plot_params_and_markers_default.copy()
    plot_params_and_markers_dict.update(plot_params_and_markers_user)

    if len(plot_params_and_markers_dict['unit_real_ids']) != n_neurons:
        raise ValueError(
            'length of unit_ids should be equal to number of neurons! \n'
            'Unit_Ids: ' + plot_params_and_markers_dict[
                'unit_real_ids'] + 'ungleich NumOfNeurons: ' + n_neurons)

    ax5 = plt.subplot(position[0], position[1], position[2])
    ax5.set_title('Unitary Events')
    for n in range(n_neurons):
        for tr, data_tr in enumerate(data):
            ax5.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) *
                     tr + n * (n_trail + 1) + 1,
                     ls='None', marker=plot_params_and_markers_dict[
                    'data_symbol'], markersize=plot_params_and_markers_dict[
                    'data_markersize'], color=plot_params_and_markers_dict[
                    'data_markercolor'][0])
            # TODO: rename sig_idx_win to signum_index_window?
            sig_idx_win = numpy.where(
                joint_suprise_dict['Js'] >= joint_suprise_significance)[0]
            if len(sig_idx_win) > 0:
                # TODO: rename x and xx to be self-explaining
                x = numpy.unique(
                    joint_suprise_dict['indices']['trial' + str(tr)])
                if len(x) > 0:
                    xx = []
                    for j in sig_idx_win:
                        xx = numpy.append(xx, x[numpy.where(
                            (x * binsize >= t_winpos[j]) &
                            (x * binsize < t_winpos[j] + window_size))])
                    ax5.plot(
                        numpy.unique(xx) * binsize,
                        numpy.ones_like(numpy.unique(xx)) * tr + n * (
                            n_trail + 1) + 1,
                        ms=plot_params_and_markers_dict['event_markersize'],
                        marker=plot_params_and_markers_dict['event_symbol'],
                        ls='', markerfacecolor=plot_params_and_markers_dict
                        ['event_markerfacecolor'],
                        markeredgecolor=plot_params_and_markers_dict
                        ['event_markeredgecolor'])

        if n < n_neurons - 1:
            ax5.axhline((tr + 2) * (n + 1), lw=plot_params_and_markers_dict[
                'lw'], color='b')
    ax5.set_xlim((min(t_winpos) - window_size).rescale('ms').magnitude,
                 (max(t_winpos) + window_size).rescale('ms').magnitude)
    ax5.set_ylim(0, (tr + 2) * (n + 1) + 1)

    if plot_params_and_markers_dict['boolean_vertical_lines']:
        x_line_vertical = MultipleLocator(
            plot_params_and_markers_dict['major_tick_width_time']).tick_values(
            t_start.magnitude, t_stop.magnitude)
        for xc in x_line_vertical:
            ax5.axvline(xc, lw=plot_params_and_markers_dict['lw'],
                        color=plot_params_and_markers_dict
                        ['color_vertical_lines'])

    ax5.xaxis.set_major_locator(
        MultipleLocator(plot_params_and_markers_dict['major_tick_width_time']))
    ax5.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax5.xaxis.set_minor_locator(
        MultipleLocator(plot_params_and_markers_dict['major_tick_width_time'] /
                        (plot_params_and_markers_dict
                         ['number_minor_ticks_time'] + 1)))
    # set y-axis
    y_ticks_list = []
    # finding y-tick position for trail no.1
    for yt1 in range(1, n_neurons * n_trail, n_trail + 1):
        y_ticks_list.append(yt1)
    # finding y-tick position for trail interval of 15
    for n in range(n_neurons):
        for yt2 in range(n * (n_trail + 1) + 15, (n + 1) * n_trail, 15):
            y_ticks_list.append(yt2)
    y_ticks_list.sort()

    y_ticks_labels_list = [1]
    # setting y-tick interval to 15 trails
    number_of_y_ticks_per_neuron = math.floor(n_trail / 15)
    for i in range(number_of_y_ticks_per_neuron):
        y_ticks_labels_list.append((i + 1) * 15)

    auxiliary_list = y_ticks_labels_list
    # adding n_neuron times the y_ticks_labels_list to itself, so that each
    # neuron has the same y_ticks_labels
    for i in range(n_neurons - 1):
        y_ticks_labels_list += auxiliary_list

    ax5.set_yticks(y_ticks_list)
    ax5.set_yticklabels(y_ticks_labels_list,
                        fontsize=plot_params_and_markers_dict['fsize'])

    ax5.set_xlabel('Time [ms]', fontsize=plot_params_and_markers_dict['fsize'])
    ax5.set_ylabel('Trial', fontsize=plot_params_and_markers_dict['fsize'])


def plot_unitary_events_simplified(
        data, joint_suprise_dict, joint_suprise_significance, binsize,
        window_size, window_step, n_neurons):
    """
    Parameters
    ----------
    data: list of spiketrains
        list of spiketrains in different trials as representation of
        neural activity
    joint_suprise_dict: dictionary
        JointSuprise dictionary
    joint_suprise_significance: list of floats
        list of suprise measure
    binsize: Quantity scalar with the dimension time
       size of bins for descritizing spike trains
    window_size: Quantity scalar with dimension time
       size of the window of analysis
    window_step: Quantity scalar with dimension time
       size of the window step
    n_neurons: integer
        number of Neurons
    plot_params_and_markers_user: dictionary
        plotting parameters and marker properties from the user
    position: tuple
        pos is a three integer-tuple, where the first integer is the number
        of rows, the second the number of columns, and the third the index
        of the subplot
    """
    print("old")
    print("plotting unitary_events_simplified ...")
    # t_start: start of spike recording ;; t_stop: end of spike recording
    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    # t_winpos:_winpos returns an ndarray of evenly spaced values,
    # representing the time depending position of the analysis window
    t_winpos = ue._winpos(t_start, t_stop, window_size, window_step)
    # n_trial: data-length is equivalent to the number of recorded trails
    n_trial = len(data)
    # t_start, t_stop, t_winpos: <class 'quantities.quantity.Quantity'>
    # n_trial: <class 'int'>

    ax6 = plt.subplot(7, 1, 7)
    ax6.set_title('Unitary Events (simplified)')
    for n in range(n_neurons):
        # enumerate(): takes a collection and returns it as an enumerate object
        # function adds a counter as the key of the enumerate object
        # e.g: enumerate(data) -> [(0, [<SpikeTrain1(array([ 26., ..., 1873.])
        # * ms, [0.0 ms, 2100.0 ms])> ; <SpikeTrain2(array([3., ..., 2032.])
        # * ms, [0.0 ms, 2100.0 ms])> ]
        for tr, data_tr in enumerate(data):
            # tr: counter (here: 0 to end);; data_tr: content of iterable
            # object data: list of SpikeTrain-arrays

            # plotting all spike events
            # x: data_tr[n].rescale('ms').magnitude,
            # y: numpy.ones_like(data_tr[n].magnitude) * tr +
            # n * (n_trial + 1) + 1
            # numpy.ones_like(): return an array of ones with the same shape
            # and type as the given array
            ax6.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) * tr +
                     n * (n_trial + 1) + 1,
                     ls='None', marker='.', markersize=0.5, color="k")

            # TODO: rename sig_idx_win, x and xx to be self-explaining
            # searching for unitary events
            # numpy.where(condition[x,y]): return elements chosen from x or y
            # depending on condition;; true->yield x, false->yield y
            # joint_suprise_dict['Js']: float-array
            # joint_suprise_significance: float

            sig_idx_win = numpy.where(
                joint_suprise_dict['Js'] >= joint_suprise_significance)[0]
            # sig_idx_win: array of spike-indicies, where the condition is true
            # -> calculated suprise is greater/equal than suprise-significance
            # TODO: think about using nonzero instead of where

            if len(sig_idx_win) > 0:
                # numpy.unique: find the unique elements of an array and
                # returns the sorted unique elements of an array
                # -> remove all multiple elements of the array, so that the
                # 'orginal' remains and sort it
                # joint_suprise_dict['indices']['trial23'] = [91. 91.
                # ..., 309. 309. ...] -> x = [91. 309.]
                # x: ???locations of possible significant correlation
                # between spike events???
                x = numpy.unique(
                    joint_suprise_dict['indices']['trial' + str(tr)])

                if len(x) > 0:
                    xx = []
                    for j in sig_idx_win:
                        # append from x the i-th element to xx, when the
                        # i-th_ele*binsize is in the analysis-window meaning
                        # >= t_winpos[j] AND < t_winpos[j] + window_size
                        xx = numpy.append(xx, x[numpy.where(
                            (x * binsize >= t_winpos[j]) &
                            (x * binsize < t_winpos[j] + window_size))])

                    # plotting all unitary events
                    # x: numpy.unique(xx) * binsize ;; multiplying with binsize
                    # to undo the binning, which was done before
                    # (in creating the joint_suprise_dict)
                    # y: numpy.ones_like(numpy.unique(xx)) * tr +
                    # n * (n_trial + 1) + 1
                    ax6.plot(
                        numpy.unique(xx) * binsize,
                        numpy.ones_like(numpy.unique(xx)) * tr + n * (
                                n_trial + 1) + 1,
                        ms=5, marker='s', ls='', markeredgecolor='r')

        # horizontal separation line
        if n < n_neurons - 1:
            ax6.axhline((tr + 2) * (n + 1))

        ax6.set_xlim((min(t_winpos) - window_size).rescale('ms').magnitude,
                     (max(t_winpos) + window_size).rescale('ms').magnitude)
        ax6.set_ylim(0, (tr + 2) * (n + 1) + 1)
    ax6.set_xlabel('Time [ms]')
    ax6.set_ylabel('Trial')


def _checking_user_entries_of_plot_ue(
        data, joint_suprise_dict, joint_suprise_significance, binsize,
        window_size, window_step, n_neurons, position,
        **plot_params_and_markers_user):

    if (not isinstance(data, list) and not isinstance(
            data, numpy.ndarray)):  # sollen weiter Typen erlaubt sein???
        raise TypeError('data must be a list (of spiketrains)')

    if not isinstance(joint_suprise_dict, dict):
        raise TypeError('joint_suprise_dict must be a dictionary')
    else:  # checking if all keys are correct
        if "Js" not in joint_suprise_dict:
            raise KeyError('"Js"-key is missing in joint_suprise_dict')
        if "indices" not in joint_suprise_dict:
            raise KeyError('"indices"-key is missing in joint_suprise_dict')
        if "n_emp" not in joint_suprise_dict:
            raise KeyError('"n_emp"-key is missing in joint_suprise_dict')
        if "n_exp" not in joint_suprise_dict:
            raise KeyError('"n_exp"-key is missing in joint_suprise_dict')
        if "rate_avg" not in joint_suprise_dict:
            raise KeyError('"rate_avg"-key is missing in joint_suprise_dict')

    if ((not isinstance(joint_suprise_significance, list)) and (
            not isinstance(joint_suprise_significance, numpy.float64))
            and (not isinstance(joint_suprise_significance, numpy.ndarray))):
        raise TypeError(
            'joint_suprise_significance must be a list (of floats)')
    elif isinstance(joint_suprise_significance, list):
        for i in joint_suprise_significance:
            if ((not isinstance(joint_suprise_significance[i], numpy.float64))
                    and (not isinstance(joint_suprise_significance[i], float
                                        ))):
                raise TypeError('elements of the joint_suprise_significance '
                                'list are NOT floats')

    if not isinstance(binsize, pq.quantity.Quantity):  # quantity scaler
        raise TypeError('binsize must be a quantity scaler/int')

    if not isinstance(window_size, pq.quantity.Quantity):
        raise TypeError('window_size must be a quantity scaler/int')

    if not isinstance(window_step, pq.quantity.Quantity):
        raise TypeError('window_step must be a quantity scaler/int')

    if not isinstance(n_neurons, int):
        raise TypeError('n_neurons must be an integer')

    if len(position) != 6:
        raise ValueError('position must have 6 tuples to cover all subplots')

    if not isinstance(plot_params_and_markers_user, dict):
        raise TypeError('plot_params_user must be a dictionary')
