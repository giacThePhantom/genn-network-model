"""
Make plots. TODO cleanup
"""

from asyncio import protocols
import math
from pathlib import Path
import pickle
from pprint import pprint
from typing import Dict, List, Tuple

from matplotlib.patches import Rectangle
import numpy as np
import tables

from simulation import TestFirstProtocol, FirstProtocol
from protocol import Protocol
from third_protocol import ThirdProtocol
from matplotlib import cm, pyplot as plt

from reading_parameters import get_argparse_template, get_parameters, parse_cli

def make_sdf(spike_times: np.ndarray, spike_ids, n_all_ids, dt, sigma):
    """
    Calculate the spike density function of a train of spikes. This works by convolving the signal
    with a gaussian kernel.

    Arguments
    ---------
    spike_times
        the times at which spikes were recorded
    spike_ids
        the ids of the neurons that fired at the corresponding time
    n_all_ids
        the number of distinct ids
    dt
        how long a spike should be (in ms) - used to define the granularity of the gaussian kernel
    sigma
        the standard deviation of the gaussian kernel.

    Returns
    -------
    A pair. The first element is a matrix of activation patterns, in which the rows are sorted by time
    (starting from t_min - 3sigma and ending at t_max + 3sigma), and the columns are indexed by neuron id.
    The second element is the list of timesteps.
    """

    t0 = spike_times[0]
    tmax = spike_times[-1]
    tleft = t0-3*sigma
    tright = tmax+3*sigma
    n = int((tright-tleft)/dt)
    sdfs = np.zeros((n, n_all_ids))
    kwdt = 3*sigma
    i = 0
    x = np.arange(-kwdt, kwdt, dt)
    x = np.exp(-np.power(x, 2)/(2*sigma*sigma))
    x = x/(sigma*np.sqrt(2.0*np.pi))*1000.0
    if spike_times is not None:
        for t, sid in zip(spike_times, spike_ids):
            sid = int(sid)
            if (t > t0 and t < tmax):
                left = int((t-tleft-kwdt)/dt)
                right = int((t-tleft+kwdt)/dt)
                if right <= n:
                    sdfs[left:right, sid] += x

    return sdfs

def glo_avg(sdf: np.ndarray, n):
    # get the average activation intensity across n-sized groups of glomeruli
    # for each timestep
    nglo= sdf.shape[1]//n
    gsdf= np.zeros((sdf.shape[0],nglo))
    for i in range(nglo):
        gsdf[:,i]= np.mean(sdf[:,n*i:n*(i+1)],axis=1)
    return gsdf

def parse_data(exp_name, vars_to_read: Dict[str, List[str]]) -> Tuple[Dict[str, np.ndarray], Protocol]:
    dirpath = Path("/media/data/thesis_output") / exp_name
    data = tables.open_file(str(dirpath / "tracked_vars.h5"))
    with (dirpath / "protocol.pickle").open("rb") as f:
        protocol: TestFirstProtocol = pickle.load(f)

    # TODO: one could actually just return the table and access everything
    # via absolute paths like /orn/ra
    to_return = {}
    for pop_name, vars in vars_to_read.items():
        for var in vars:
            to_return[f"{pop_name}_{var}"] = data.root[pop_name][var]

    return to_return, protocol


def subplot_spike_pane(spike_t, spike_ID, idx, factor, toff, tend, ax):
    #fil1 = spike_ID >= idx * n
    #fil2 = spike_ID < (idx+1) * n

    #st = spike_t[np.logical_and(fil1, fil2)]
    st = spike_t[spike_ID == idx * factor]
    st = st[st >= toff]
    st = st[st <= toff+tend]
    #st = st[::10]
    st = np.unique(st) # wtf?
    st = np.reshape(st, (1, -1))
    x = np.vstack((st, st))
    y = np.ones(x.shape)
    y[0, :] = -70.0
    y[1, :] = 20.0
    ax.plot(x, y, 'k', lw=0.2)

def subplot_smoothed(t, data, ax, k):
    kernel = np.ones((k,))/k
    convolved = np.convolve(data, kernel, mode='same')
    ax.plot(t, convolved, 'k', linewidth=0.5)


def plot_spikes(param, exp_name, **kwargs):
    to_read =  {
        "or": ["ra"],
        "orn": ["V", "spikes"],
        "pn": ["V", "spikes"],
        "ln": ["V", "spikes"]
    }

    precision = kwargs.get("precision", None)

    data, protocol = parse_data(exp_name, to_read)

    plt.rc('font', size=8)
    ra = data["or_ra"]

    # select a timestep
    # dt = protocol.param["simulation"]["dt"]
    dt = 0.2
    if precision is None:
        precision = dt
    scale_up = int(precision // dt)
    ra_times = ra[:, 0]
    is_first_protocol = isinstance(protocol, FirstProtocol)
    step = 1 if is_first_protocol else 2

    for i in range(0, len(protocol.events), step):
        cur_step = protocol.events[i]

        t_off = cur_step["t_start"]
        t_end = cur_step["t_end"] + protocol.resting_duration
        t = np.arange(t_off, t_end, dt)

        if is_first_protocol:
            odor_name = cur_step["odor_name"]
            concentration = cur_step["concentration"]
            name = f"{odor_name}_{concentration:.1g}"
        else:
            od1 = f'{cur_step["odor_name"]}_{cur_step["concentration"]:.1g}'
            step2 = protocol.events[i+1]
            od2 = f'{step2["odor_name"]}_{step2["concentration"]:.1g}'
            name = f"{od1}-vs-{od2}"



        figure, ax = plt.subplots(4, sharex=True, layout="constrained")
        figure.suptitle(name, fontsize=14)
        # Pick the strongest glomerulus at the current experiment

        # First, pick the right time
        start_ra = np.searchsorted(ra_times, t_off)
        end_ra = np.searchsorted(ra_times, t_end, 'right')
        end_ra -= 1

        # the last event may have less data
        if end_ra - start_ra < len(t):
            t = t[:end_ra - start_ra]
        ra_vals = ra[start_ra:end_ra, 1:]
        # Then, sum the overall activation per neuron
        ra_sum = np.sum(ra_vals, axis=0)
        ra_most_active = np.argmax(ra_sum)

        # plot the most active neuron over time
        ax[0].set_title(f"OR best ra activation rate (of neuron {ra_most_active})")
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].plot(t, ra_vals[:,ra_most_active], 'k', linewidth=0.5)
        ax[0].set_xlim([t[0], t[-1]])
        ax[0].set_ylim([-0.025, 1.5*np.amax(ra_vals[:, ra_most_active])])

        ax[0].add_patch(
            Rectangle((t_off + 100, -0.025), protocol.event_duration, 0.2*np.amax(ra_vals[:, ra_most_active]),
                    edgecolor='grey',
                    facecolor='grey',
                    fill=True)
        )

        or_n = param["neuron_populations"]["or"]["n"]
        for j, pop in enumerate(to_read):
            if j == 0:
                continue # exclude OR

            to_plot = data[f"{pop}_V"][start_ra:end_ra, 1:]
            pop_n = param["neuron_populations"][pop]["n"]
            pop_spikes = data[f"{pop}_spikes"]
            pop_spikes_t = pop_spikes[:, 0]
            pop_spikes_ids = pop_spikes[:, 1]

            factor = pop_n // or_n
            ax[j].set_title(f"{pop.upper()} V for neuron {ra_most_active*factor}")
            ax[j].spines['right'].set_visible(False)
            ax[j].spines['top'].set_visible(False)
            subplot_smoothed(t, to_plot[:, ra_most_active], ax[j], scale_up)
            subplot_spike_pane(pop_spikes_t, pop_spikes_ids,
                            ra_most_active, factor, t_off, t_end, ax[j])
            ax[j].set_ylim([-90, 40])
            ax[j].set_ylabel("mV")

        if is_first_protocol:
            odor_name = cur_step["odor_name"]
            concentration = cur_step["concentration"]
            name = f"{odor_name}_{concentration:.1g}"
        else:
            od1 = f'{cur_step["odor_name"]}_{cur_step["concentration"]:.1g}'
            step2 = protocol.events[i+1]
            od2 = f'{step2["odor_name"]}_{step2["concentration"]:.1g}'
            name = f"{od1}-vs-{od2}"
        filename = f"{param['simulation']['simulation']['output_path']}/{exp_name}/{exp_name}_raw_spikes_{i}_{name}.png"
        print(f"saving to {filename}")
        plt.savefig(filename, dpi=1000)
        print(f"saved")




def plot_heatmap(param, exp_name, **kwargs):
    data, protocol = parse_data(
        exp_name, {
            "orn": ["spikes"], "pn": ["spikes"], "ln": ["spikes"]
        }
    )

    plt.rc("font", size=8)
    plt.rcParams['text.usetex'] = True
    n_or = param["neuron_populations"]["or"]["n"]

    if isinstance(protocol, ThirdProtocol):
        # select:
        #   pure Geo [1e-6, 1e-5, 1e-4, 1e-3] OR
        #   pure IAA [1e-3, 1e-1] OR
        #   mixed IAA [(0.001, 0.001)]
        iis = [2, 4, 6, 8, 10, 20, 18]
        titles = []
        for i in range(-6, -2):
            titles.append("Geo, $c = 10^{%d}$" % i)
        for i in [-3, -1]:
            titles.append("IAA, $c = 10^{%d}$" % i)
        titles.append("IAA+Geo, $c = 10^{%d}$" % -3)

    else:
        # TODO
        iis = list(range(10))

    for pop in ["orn", "pn", "ln"]:
        sdfs = []
        glo_avg_sdfs = []
        for i in iis:
            od_active = protocol.events[i]
            # FIXME
            od_inactive = protocol.events[i]
            od_active_tstart = od_active["t_start"]
            od_inactive_tend = od_inactive["t_end"]
            #print(od_active_tstart, od_inactive_tend)


            pop_spikes = data[f"{pop}_spikes"]
            pop_spikes_t = pop_spikes[:, 0]
            pop_spikes_ids = pop_spikes[:, 1]

            left_spikes_active_idx = np.searchsorted(pop_spikes_t, od_active_tstart)
            right_spikes_inactive_idx = np.searchsorted(pop_spikes_t, od_inactive_tend, 'right')
            n_pop = param["neuron_populations"][pop]["n"]
            factor = n_pop // n_or

            spikes_t = pop_spikes_t[left_spikes_active_idx:right_spikes_inactive_idx]
            spikes_ids = pop_spikes_ids[left_spikes_active_idx:right_spikes_inactive_idx]

            if len(spikes_t) == 0:
                break

            sigma_sdf = 100.0
            dt_sdf = 1.0

            sdfs.append(make_sdf(spikes_t,
                        spikes_ids, n_pop, dt_sdf, sigma_sdf))

            glo_avg_sdfs.append(glo_avg(sdfs[-1], factor))

        min_cbar = -20
        max_cbar = 100

        fig, ax = plt.subplots(1, len(iis), sharey=True, layout="constrained", figsize=(10,4))
        fig.suptitle(f"{protocol.param['connectivity_type']} configuration in {pop.upper()}")
        fig.text(0.5, 0.01, "Time ($s$)", ha='center')
        ax[0].set_ylabel("Neuron group")
        for i in range(len(iis)):
            ax[i].set_title(titles[i])
            last_image = ax[i].imshow(glo_avg_sdfs[i].T, vmin=min_cbar, vmax=max_cbar, cmap="hot")
            ax[i].set_aspect(60)
            ax[i].set_xticks(np.linspace(0, od_inactive_tend - od_active_tstart, 3))
        cbar = fig.colorbar(last_image, ax=ax[i])
        cbar.ax.set_ylabel("SDF ($Hz$)")
        filename = f"{exp_name}_heatmap_{pop}.png"
        plt.savefig(filename, dpi=700, bbox_inches='tight')
        plt.cla()
        plt.clf()

        fig, ax = plt.subplots()

        fig.suptitle(f"Mean activation of {pop} glomeruli neurons over time")
        for idx in range(len(iis)):
            # take the most active glomerulus and plot its activation over time
            most_active_glo = np.argmax(np.mean(glo_avg_sdfs[idx], axis=0))
            ax.plot(glo_avg_sdfs[idx][:, most_active_glo], label=titles[idx])
            ax.set_ylabel("SDF ($Hz$)")
            ax.set_xlabel("Time ($s$)")
        ax.set_xticks(np.linspace(0, od_inactive_tend - od_active_tstart, 3))
        ax.legend()
        filename = f"{exp_name}_lines_{pop}.png"
        plt.savefig(filename, dpi=700, bbox_inches='tight')
        plt.cla()
        plt.clf()


def plot_sdf_over_c(param, exp_name, **kwargs):
    data, protocol = parse_data(
        exp_name, {
            "orn": ["spikes"], "pn": ["spikes"], "ln": ["spikes"]
        }
    )

    plt.rc("font", size=8)
    plt.rcParams['text.usetex'] = True
    n_or = param["neuron_populations"]["or"]["n"]

    if not isinstance(protocol, FirstProtocol):
        raise ValueError("This plot requires FirstProtocol")


    for pop in ["orn"]:
        concentrations = {"iaa": [0, 1, 2], "geo": [3, 4, 5]}
        for odor, ii in concentrations.items():
            max_sdfs = []
            max_concentrations = []
            for i in ii:
                od_active = protocol.events[i]
                # FIXME
                od_inactive = protocol.events[i]
                od_active_tstart = od_active["t_start"]
                od_inactive_tend = od_inactive["t_end"]

                pop_spikes = data[f"{pop}_spikes"]
                pop_spikes_t = pop_spikes[:, 0]
                pop_spikes_ids = pop_spikes[:, 1]

                left_spikes_active_idx = np.searchsorted(pop_spikes_t, od_active_tstart)
                right_spikes_inactive_idx = np.searchsorted(pop_spikes_t, od_inactive_tend, 'right')
                n_pop = param["neuron_populations"][pop]["n"]
                factor = n_pop // n_or

                spikes_t = pop_spikes_t[left_spikes_active_idx:right_spikes_inactive_idx]
                spikes_ids = pop_spikes_ids[left_spikes_active_idx:right_spikes_inactive_idx]

                sigma_sdf = 100.0
                dt_sdf = 1.0
                n_pop = param["neuron_populations"][pop]["n"]
                factor = n_pop // n_or

                sdf = make_sdf(spikes_t, spikes_ids, n_pop, dt_sdf, sigma_sdf)
                avgd = glo_avg(sdf, factor)

                max_sdfs.append(np.max(avgd))
                max_concentrations.append(od_active["concentration"])


            print(max_concentrations)
            plt.plot(max_concentrations, max_sdfs, label=odor)
        plt.legend()
        plt.show()


        if len(spikes_t) == 0:
            break

        sigma_sdf = 100.0
        dt_sdf = 1.0

def plot_mono(params, exp_name, **kwargs):
    pass

if __name__ == "__main__":
    parser = get_argparse_template()
    plot_group = parser.add_argument_group("plot")
    plot_group.add_argument("plot", choices=['spikes', 'heatmap', 'sdf-over-c', 'mono'])
    plot_group.add_argument("--precision", help="For spikes plotting, set the desired resolution (in ms). Defaults to the simulation dt.")
    params = parse_cli(parser)
    name = params['simulation']['name']

    print(params["cli"])

    match params["cli"]["plot"]:
        case "spikes":
            f = plot_spikes
        case "heatmap":
            f = plot_heatmap
        case "sdf-over-c":
            f = plot_sdf_over_c
        case "mono":
            f = plot_mono

    f(params, name, precision=params["cli"]["precision"])
