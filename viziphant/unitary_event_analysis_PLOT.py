import math
import numpy
import quantities as pq
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
import elephant.unitary_event_analysis as ue

plot_params_default = {
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
    'color_vertical_lines': "r"
}

plot_markers_default = {
    'data_symbol': ".",
    'data_markersize': 0.5,
    'data_markercolor': ("k"),
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
        window_size, window_step, n_neurons, plot_params_user,
        plot_markers_user, position):
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
    pattern_hash: list of integers
       list of interested patterns in hash values
       (see hash_from_pattern and inverse_hash_from_pattern functions)
    n_neurons: integer
        number of Neurons
    plot_params_user: dictionary
        plotting parameters from the user
    plot_markers_user: list of dictionaries
        marker properties from the user
    position: list of position-tupels
        (posSpikeEvents(c,r,i), posSpikeRates(c,r,i),
         posCoincidenceEvents(c,r,i), posCoincidenceRates(c,r,i),
        posStatisticalSignificance(c,r,i), posUnitaryEvents(c,r,i))
        pos is a three integer-tupel, where the first integer is the number
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
        _checkungUserEntries_plot_UE(data, joint_suprise_dict, 
        joint_suprise_significance, binsize,
        window_size, window_step, n_neurons, plot_params_user, 
        plot_markers_user, position)
    except (TypeError, KeyError) as errors:
        print(errors)
        raise errors
    """

    # subplots format
    plot_params = plot_params_default.copy()
    plot_params.update(plot_params_user)

    if len(plot_params['unit_real_ids']) != n_neurons:
        raise ValueError(
            'length of unit_ids should be equal to number of neurons! \n'
            'Unit_Ids: ' + plot_params['unit_real_ids']
            + 'ungleich NumOfNeurons: ' + n_neurons)

    if 'suptitle' in plot_params.keys():
        plt.suptitle("Trial aligned on " +
                     plot_params['suptitle'], fontsize=20)
    plt.subplots_adjust(hspace=plot_params['hspace'],
                        wspace=plot_params['wspace'])

    plot_spike_events(
        data, window_size, window_step, n_neurons, plot_params_user,
        plot_markers_user[0], position[0])

    plot_spike_rates(
        data, joint_suprise_dict, window_size, window_step, n_neurons,
        plot_params_user, plot_markers_user[1], position[1])

    plot_coincidence_events(
        data, joint_suprise_dict, binsize, window_size, window_step, n_neurons,
        plot_params_user, plot_markers_user[2], position[2])

    plot_coincidence_rates(
        data, joint_suprise_dict, window_size, window_step, n_neurons,
        plot_params_user, plot_markers_user[3], position[3])

    plot_statistical_significance(
        data, joint_suprise_dict, joint_suprise_significance, window_size,
        window_step, n_neurons, plot_params_user, plot_markers_user[4],
        position[4])

    plot_unitary_events(
        data, joint_suprise_dict, joint_suprise_significance, binsize,
        window_size, window_step, n_neurons, plot_params_user,
        plot_markers_user[5], position[5])


def plot_spike_events(
        data, window_size, window_step, n_neurons, plot_params_user,
        plot_markers_user, position):
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
    plot_params_user: dictionary
        plotting parameters from the user
    plot_markers_user: list of dictionaries
        marker properties from the user
    position: position-tupel
        position is a three integer-tupel, where the first integer is the
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
    num_tr = len(data)

    # subplot format
    plot_params = plot_params_default.copy()
    plot_params.update(plot_params_user)
    # marker format
    plot_markers = plot_markers_default.copy()
    plot_markers.update(plot_markers_user)

    ax0 = plt.subplot(position[0], position[1], position[2])
    ax0.set_title('Spike Events')
    for n in range(n_neurons):
        for tr, data_tr in enumerate(data):
            ax0.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) * tr + n * (
                num_tr + 1) + 1, ls='none',
                marker=plot_markers['data_symbol'],
                color=plot_markers['data_markercolor'][0],
                markersize=plot_markers['data_markersize'])
        if n < n_neurons - 1:
            ax0.axhline((tr + 2) * (n + 1), lw=plot_params['lw'], color='b')

    ax0.set_xlim(0, (max(t_winpos) + window_size).rescale(
        'ms').magnitude)  # better minimum
    ax0.set_ylim(0, (tr + 2) * (n + 1) + 1)

    if (plot_params['boolean_vertical_lines']):
        x_line_vertical = MultipleLocator(
            plot_params['major_tick_width_time']).tick_values(
            t_start.magnitude, t_stop.magnitude)
        for xc in x_line_vertical:
            ax0.axvline(xc, lw=plot_params['lw'],
                        color=plot_params['color_vertical_lines'])

    ax0.xaxis.set_major_locator(
        MultipleLocator(plot_params['major_tick_width_time']))
    ax0.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax0.xaxis.set_minor_locator(
        MultipleLocator(plot_params['major_tick_width_time'] /
                        (plot_params['number_minor_ticks_time'] + 1)))
    # set y-axis
    y_ticks_list = []
    for yt1 in range(1, n_neurons * num_tr, num_tr + 1):
        y_ticks_list.append(yt1)
    for n in range(n_neurons):
        for yt2 in range(n * (num_tr + 1) + 15, (n + 1) * num_tr, 15):
            y_ticks_list.append(yt2)
    y_ticks_list.sort()

    y_ticks_labels_list = [1]
    number_of_y_ticks_per_neuron = math.floor(num_tr / 15)
    for i in range(number_of_y_ticks_per_neuron):
        y_ticks_labels_list.append((i + 1) * 15)

    auxiliary_list = y_ticks_labels_list
    for i in range(n_neurons - 1):
        y_ticks_labels_list += auxiliary_list
    # print(yticks_list)
    ax0.set_yticks(y_ticks_list)
    ax0.set_yticklabels(y_ticks_labels_list, fontsize=plot_params['fsize'])

    x_lim = ax0.get_xlim()
    ax0.text(x_lim[1], num_tr * 2 + 7, 'Neuron 2')
    ax0.text(x_lim[1], -12, 'Neuron 1')

    ax0.set_xlabel('Time [ms]', fontsize=plot_params['fsize'])
    ax0.set_ylabel('Trial', fontsize=plot_params['fsize'])


def plot_spike_rates(
        data, joint_suprise_dict, window_size, window_step, n_neurons,
        plot_params_user, plot_markers_user, position):
    """
    Visualization of the spike rates of the Unitary Event Analysis.

    The spike rates represent the number of spikes in a defined time interval.

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
    plot_params_user: dictionary
        plotting parameters from the user
    plot_markers_user: list of dictionaries
        marker properties from the user
    position: position-tupel
        pos is a three integer-tupel, where the first integer is the number
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

    # subplot format
    plot_params = plot_params_default.copy()
    plot_params.update(plot_params_user)
    # marker format
    plot_markers = plot_markers_default.copy()
    plot_markers.update(plot_markers_user)

    ax1 = plt.subplot(position[0], position[1], position[2])
    ax1.set_title('Spike Rates')
    for n in range(n_neurons):
        ax1.plot(t_winpos + window_size / 2.,
                 joint_suprise_dict['rate_avg'][:, n].rescale('Hz'),
                 label='Neuron ' + str(plot_params['unit_real_ids'][n]),
                 color=plot_markers['data_markercolor'][n],
                 lw=plot_params['lw'])

    ax1.set_xlim(0, (max(t_winpos) + window_size).rescale('ms').magnitude)
    max_val_psth = 40
    ax1.set_ylim(0, max_val_psth)

    if (plot_params['boolean_vertical_lines']):
        x_line_vertical = MultipleLocator(
            plot_params['major_tick_width_time']).tick_values(
            t_start.magnitude, t_stop.magnitude)
        for xc in x_line_vertical:
            ax1.axvline(xc, lw=plot_params['lw'],
                        color=plot_params['color_vertical_lines'])

    ax1.xaxis.set_major_locator(
        MultipleLocator(plot_params['major_tick_width_time']))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax1.xaxis.set_minor_locator(
        MultipleLocator(plot_params['major_tick_width_time'] /
                        (plot_params['number_minor_ticks_time'] + 1)))
    ax1.set_yticks([0, int(max_val_psth / 2), int(max_val_psth)])

    ax1.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    ax1.set_xlabel('Time [ms]', fontsize=plot_params['fsize'])
    ax1.set_ylabel('(1/s)', fontsize=plot_params['fsize'])


def plot_coincidence_events(
        data, joint_suprise_dict, binsize, window_size, window_step, n_neurons,
        plot_params_user, plot_markers_user, position):
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
    plot_params_user: dictionary
        plotting parameters from the user
    plot_markers_user: list of dictionaries
        marker properties from the user
    position: position-tupel
        pos is a three integer-tupel, where the first integer is the number
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
    num_tr = len(data)

    # subplot format
    plot_params = plot_params_default.copy()
    plot_params.update(plot_params_user)
    # marker format
    plot_markers = plot_markers_default.copy()
    plot_markers.update(plot_markers_user)

    ax2 = plt.subplot(position[0], position[1], position[2])
    ax2.set_title('Coincidence Events')
    for n in range(n_neurons):
        for tr, data_tr in enumerate(data):
            ax2.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) *
                     tr + n * (num_tr + 1) + 1, ls='None',
                     marker=plot_markers['data_symbol'],
                     markersize=plot_markers['data_markersize'],
                     color=plot_markers['data_markercolor'])
            ax2.plot(numpy.unique(
                joint_suprise_dict['indices']['trial' + str(tr)]) * binsize,
                numpy.ones_like(numpy.unique(
                    joint_suprise_dict['indices']['trial' + str(tr)]))
                * tr + n * (num_tr + 1) + 1,
                ls='', ms=plot_markers['event_markersize'],
                marker=plot_markers['event_symbol'],
                markerfacecolor=plot_markers['event_markerfacecolor'],
                markeredgecolor=plot_markers['event_markeredgecolor'])
        if n < n_neurons - 1:
            ax2.axhline((tr + 2) * (n + 1), lw=plot_params['lw'], color='b')
    ax2.set_ylim(0, (tr + 2) * (n + 1) + 1)
    ax2.set_xlim(0, (max(t_winpos) + window_size).rescale('ms').magnitude)

    if (plot_params['boolean_vertical_lines']):
        x_line_veritcal = MultipleLocator(
            plot_params['major_tick_width_time']).tick_values(
            t_start.magnitude, t_stop.magnitude)
        for xc in x_line_veritcal:
            ax2.axvline(xc, lw=plot_params['lw'],
                        color=plot_params['color_vertical_lines'])

    ax2.xaxis.set_major_locator(
        MultipleLocator(plot_params['major_tick_width_time']))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax2.xaxis.set_minor_locator(
        MultipleLocator(plot_params['major_tick_width_time'] /
                        (plot_params['number_minor_ticks_time'] + 1)))
    # set y-axis
    y_ticks_list = []
    for yt1 in range(1, n_neurons * num_tr, num_tr + 1):
        y_ticks_list.append(yt1)
    for n in range(n_neurons):
        for yt2 in range(n * (num_tr + 1) + 15, (n + 1) * num_tr, 15):
            y_ticks_list.append(yt2)
    y_ticks_list.sort()

    y_ticks_labels_list = [1]
    number_of_y_ticks_per_neuron = math.floor(num_tr / 15)
    for i in range(number_of_y_ticks_per_neuron):
        y_ticks_labels_list.append((i + 1) * 15)

    auxiliary_list = y_ticks_labels_list
    for i in range(n_neurons - 1):
        y_ticks_labels_list += auxiliary_list
    ax2.set_yticks(y_ticks_list)
    ax2.set_yticklabels(y_ticks_labels_list, fontsize=plot_params['fsize'])

    ax2.set_xlabel('Time [ms]', fontsize=plot_params['fsize'])
    ax2.set_ylabel('Trial', fontsize=plot_params['fsize'])


def plot_coincidence_rates(
        data, joint_suprise_dict, window_size, window_step, n_neurons,
        plot_params_user, plot_markers_user, position):
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
    plot_params_user: dictionary
        plotting parameters from the user
    plot_markers_user: list of dictionaries
        marker properties from the user
    position: position-tupel
        pos is a three integer-tupel, where the first integer is the number
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
    num_tr = len(data)

    # subplot format
    plot_params = plot_params_default.copy()
    plot_params.update(plot_params_user)
    # marker format
    plot_markers = plot_markers_default.copy()
    plot_markers.update(plot_markers_user)

    if len(plot_params['unit_real_ids']) != n_neurons:
        raise ValueError(
            'length of unit_ids should be equal to number of neurons! \n'
            'Unit_Ids: ' + plot_params[
                'unit_real_ids'] + 'ungleich NumOfNeurons: ' + n_neurons)

    ax3 = plt.subplot(position[0], position[1], position[2])
    ax3.set_title('Coincidence Rates')
    ax3.plot(t_winpos + window_size / 2.,
             joint_suprise_dict['n_emp'] / (
                 window_size.rescale('s').magnitude * num_tr),
             label='empirical', lw=plot_params['lw'],
             color=plot_markers['data_markercolor'][0])
    ax3.plot(t_winpos + window_size / 2.,
             joint_suprise_dict['n_exp'] / (
                 window_size.rescale('s').magnitude * num_tr),
             label='expected', lw=plot_params['lw'],
             color=plot_markers['data_markercolor'][1])
    ax3.set_xlim(0, (max(t_winpos) + window_size).rescale('ms').magnitude)

    if (plot_params['boolean_vertical_lines']):
        x_line_vertical = MultipleLocator(
            plot_params['major_tick_width_time']).tick_values(
            t_start.magnitude, t_stop.magnitude)
        for xc in x_line_vertical:
            ax3.axvline(xc, lw=plot_params['lw'],
                        color=plot_params['color_vertical_lines'])

    ax3.xaxis.set_major_locator(
        MultipleLocator(plot_params['major_tick_width_time']))
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax3.xaxis.set_minor_locator(
        MultipleLocator(plot_params['major_tick_width_time'] /
                        (plot_params['number_minor_ticks_time'] + 1)))
    y_ticks = ax3.get_ylim()
    ax3.set_yticks([0, y_ticks[1] / 2, y_ticks[1]])

    ax3.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    ax3.set_xlabel('Time [ms]', fontsize=plot_params['fsize'])
    ax3.set_ylabel('(1/s)', fontsize=plot_params['fsize'])


def plot_statistical_significance(
        data, joint_suprise_dict, joint_suprise_significance, window_size,
        window_step, n_neurons, plot_params_user, plot_markers_user, position):

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
    plot_params_user: dictionary
        plotting parameters from the user
    plot_markers_user: list of dictionaries
        marker properties from the user
    position: position-tupel
        pos is a three integer-tupel, where the first integer is the number
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

    # figure format
    plot_params = plot_params_default.copy()
    plot_params.update(plot_params_user)
    # marker format
    plot_markers = plot_markers_default.copy()
    plot_markers.update(plot_markers_user)

    if len(plot_params['unit_real_ids']) != n_neurons:
        raise ValueError('length of unit_ids should be equal to '
                        'number of neurons! \nUnit_Ids: ' +
                        plot_params['unit_real_ids'] +
                        'ungleich NumOfNeurons: ' +n_neurons)

    ax4 = plt.subplot(position[0], position[1], position[2])
    ax4.set_title('Statistical Significance')
    ax4.plot(t_winpos + window_size / 2., joint_suprise_dict['Js'],
             lw=plot_params['lw'],
             color=plot_markers['data_markercolor'][0])
    ax4.set_xlim(0, (max(t_winpos) + window_size).rescale('ms').magnitude)
    ax4.set_ylim(plot_params['S_ylim'])

    ax4.axhline(joint_suprise_significance, ls='-',
                color=plot_markers['data_markercolor'][1])
    ax4.axhline(-joint_suprise_significance, ls='-',
                color=plot_markers['data_markercolor'][2])
    ax4.text(t_winpos[30], joint_suprise_significance + 0.3, '$\\alpha +$',
             color=plot_markers['data_markercolor'][1])
    ax4.text(t_winpos[30], -joint_suprise_significance - 0.5, '$\\alpha -$',
             color=plot_markers['data_markercolor'][2])

    if (plot_params['boolean_vertical_lines']):
        x_line_vertical = MultipleLocator(
            plot_params['major_tick_width_time']).tick_values(
            t_start.magnitude, t_stop.magnitude)
        for xc in x_line_vertical:
            ax4.axvline(xc, lw=plot_params['lw'],
                        color=plot_params['color_vertical_lines'])

    ax4.xaxis.set_major_locator(
        MultipleLocator(plot_params['major_tick_width_time']))
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax4.xaxis.set_minor_locator(
        MultipleLocator(plot_params['major_tick_width_time'] /
                        (plot_params['number_minor_ticks_time'] + 1)))
    ax4.set_yticks([ue.jointJ(0.99), ue.jointJ(0.5), ue.jointJ(0.01)])

    ax4.set_xlabel('Time [ms]', fontsize=plot_params['fsize'])
    ax4.set_yticklabels([alpha + 0.5, alpha, alpha - 0.5])


def plot_unitary_events(
        data, joint_suprise_dict, joint_suprise_significance, binsize,
        window_size, window_step, n_neurons, plot_params_user,
        plot_markers_user, position):

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
    plot_params_user: dictionary
        plotting parameters from the user
    plot_markers_user: list of dictionaries
        marker properties from the user
    position: position-tupel
        pos is a three integer-tupel, where the first integer is the number
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
    num_tr = len(data)

    # subplot format
    plot_params = plot_params_default.copy()
    plot_params.update(plot_params_user)
    # marker format
    plot_markers = plot_markers_default.copy()
    plot_markers.update(plot_markers_user)

    if len(plot_params['unit_real_ids']) != n_neurons:
        raise ValueError(
            'length of unit_ids should be equal to number of neurons! \n'
            'Unit_Ids: ' + plot_params[
                'unit_real_ids'] + 'ungleich NumOfNeurons: ' + n_neurons)

    ax5 = plt.subplot(position[0], position[1], position[2])
    ax5.set_title('Unitary Events')
    for n in range(n_neurons):
        for tr, data_tr in enumerate(data):
            ax5.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) *
                     tr + n * (num_tr + 1) + 1,
                     ls='None', marker=plot_markers['data_symbol'],
                     markersize=plot_markers['data_markersize'],
                     color=plot_markers['data_markercolor'])
            sig_idx_win = numpy.where(
                joint_suprise_dict['Js'] >= joint_suprise_significance)[0]
            if len(sig_idx_win) > 0:
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
                            num_tr + 1) + 1,
                        ms=plot_markers['event_markersize'],
                        marker=plot_markers['event_symbol'], ls='',
                        markerfacecolor=plot_markers['event_markerfacecolor'],
                        markeredgecolor=plot_markers['event_markeredgecolor'])

        if n < n_neurons - 1:
            ax5.axhline((tr + 2) * (n + 1), lw=plot_params['lw'], color='b')
    ax5.set_xlim(0, (max(t_winpos) + window_size).rescale('ms').magnitude)
    ax5.set_ylim(0, (tr + 2) * (n + 1) + 1)

    if (plot_params['boolean_vertical_lines']):
        x_line_vertical = MultipleLocator(
            plot_params['major_tick_width_time']).tick_values(
            t_start.magnitude, t_stop.magnitude)
        for xc in x_line_vertical:
            ax5.axvline(xc, lw=plot_params['lw'],
                        color=plot_params['color_vertical_lines'])

    ax5.xaxis.set_major_locator(
        MultipleLocator(plot_params['major_tick_width_time']))
    ax5.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax5.xaxis.set_minor_locator(
        MultipleLocator(plot_params['major_tick_width_time'] /
                        (plot_params['number_minor_ticks_time'] + 1)))
    # set y-axis
    y_ticks_list = []
    for yt1 in range(1, n_neurons * num_tr, num_tr + 1):
        y_ticks_list.append(yt1)
    for n in range(n_neurons):
        for yt2 in range(n * (num_tr + 1) + 15, (n + 1) * num_tr, 15):
            y_ticks_list.append(yt2)
    y_ticks_list.sort()

    y_ticks_labels_list = [1]
    number_of_y_ticks_per_neuron = math.floor(num_tr / 15)
    for i in range(number_of_y_ticks_per_neuron):
        y_ticks_labels_list.append((i + 1) * 15)

    auxiliary_list = y_ticks_labels_list
    for i in range(n_neurons - 1):
        y_ticks_labels_list += auxiliary_list

    ax5.set_yticks(y_ticks_list)
    ax5.set_yticklabels(y_ticks_labels_list, fontsize=plot_params['fsize'])

    ax5.set_xlabel('Time [ms]', fontsize=plot_params['fsize'])
    ax5.set_ylabel('Trial', fontsize=plot_params['fsize'])


def _checking_user_entries_of_plot_UE(
        data, joint_suprise_dict, joint_suprise_significance, binsize,
        window_size, window_step, pattern_hash, n_neurons, plot_params_user,
        plot_markers_user, position):

    if (not isinstance(data, list) and not isinstance(
            data, numpy.ndarray)):  # sollen weiter Typen erlaubt sein???
        raise TypeError('data must be a list (of spiketrains)')

    if (not isinstance(joint_suprise_dict, dict)):
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
    elif (isinstance(joint_suprise_significance, list)):
        for i in joint_suprise_significance:
            if ((not isinstance(joint_suprise_significance[i], numpy.float64))
                and (not isinstance(joint_suprise_significance[i], float))):
                raise TypeError('elements of the joint_suprise_significance '
                                'list are NOT floats')

    if (not isinstance(binsize, pq.quantity.Quantity)):  # quantity scaler
        raise TypeError('binsize must be a quantity scaler/int')

    if (not isinstance(window_size, pq.quantity.Quantity)):
        raise TypeError('window_size must be a quantity scaler/int')

    if (not isinstance(window_step, pq.quantity.Quantity)):
        raise TypeError('window_step must be a quantity scaler/int')

    if (not isinstance(pattern_hash, list)
            and not isinstance(pattern_hash, numpy.ndarray)):
        raise TypeError('pattern_hash must be a list (of integers)')
    else:
        for i in pattern_hash:
            if (not isinstance(pattern_hash[i], int)):
                raise TypeError(
                    'elements of the pattern_hash list are NOT integers')

    if (not isinstance(n_neurons, int)):
        raise TypeError('n_neurons must be an integer')

    if (not isinstance(plot_params_user, dict)):
        raise TypeError('plot_params_user must be a dictionary')

    if ((not isinstance(plot_markers_user, list)) and (
            not isinstance(plot_markers_user, numpy.ndarray))):
        raise TypeError("plot_markers_user must be a list")
    else:
        for i in plot_markers_user:
            if (type(plot_markers_user[i] != dict)):
                raise TypeError('elements of th plot_markers_user list '
                                'are NOT dictionaries')
