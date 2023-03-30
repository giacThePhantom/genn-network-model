"""
Make plots. TODO cleanup
"""

from pathlib import Path
import pickle
from typing import Dict, List, Tuple, cast

from matplotlib.patches import Rectangle
import numpy as np
import tables

from .simulation import TestFirstProtocol, FirstProtocol
from .protocol import Protocol
from .third_protocol import ThirdProtocol
from matplotlib import pyplot as plt

from .reading_parameters import get_argparse_template, parse_cli

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
    sdfs_out = np.zeros((n, n_all_ids))
    kernel_width = 3*sigma
    kernel = np.arange(-kernel_width, kernel_width, dt)
    kernel = np.exp(-np.power(kernel, 2)/(2*sigma*sigma))
    kernel = kernel/(sigma*np.sqrt(2.0*np.pi))*1000.0
    if spike_times is not None:
        for t, sid in zip(spike_times, spike_ids):
            sid = int(sid)
            if (t > t0 and t < tmax):
                left = int((t-tleft-kernel_width)/dt)
                right = int((t-tleft+kernel_width)/dt)
                if right <= n:
                    sdfs_out[left:right, sid] += kernel

    return sdfs_out

def glo_avg(sdf: np.ndarray, n_per_glo):
    """
    Get the average activation intensity across n-sized groups of glomeruli
    for each timestep.

    Arguments
    ---------
    sdf: np.ndarray
        an SDF obtained by eg. `make_sdf`
    n: int
        How many neurons per glomerulus
    """
    n_glo = sdf.shape[1]//n_per_glo
    glo_sdfs_out= np.zeros((sdf.shape[0],n_glo))
    for i in range(n_glo):
        glo_sdfs_out[:,i]= np.mean(sdf[:,n_per_glo*i:n_per_glo*(i+1)],axis=1)
    return glo_sdfs_out

def parse_data(param, exp_name, vars_to_read: Dict[str, List[str]]) -> Tuple[Dict[str, np.ndarray], Protocol]:
    """
    Load a HDFS table with the selected variables into a dictionary.

    Arguments
    ---------
    param: dict
        the parameters
    exp_name: string
        the experiment name
    vars_to_read: dict
        a dictionary (key -> list of vars) of the variables to extract.
        Refer to `Simulation` for details
    """
    dirpath = Path(param["simulations"]['simulation']['output_path']) / exp_name
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


def subplot_spike_pane(spike_t: np.ndarray, spike_ids: np.ndarray, idx: int, factor: int, t_start: float, t_end: float, ax):
    """
    Superimpose spikes on top of of an axis. A "spike" is a dirac-like impulse going
    from -70 to 20 mV. This function is necessary as Genn truncates the at the spike event *before* logging.

    Arguments
    ---------
    spike_t:
        The spike event times
    spike_ids:
        The spike event units that fired at the corresponding time
    idx:
        The OR neuron with the strongest activation that we want to show
    factor:
        The ratio `this_pop.size / or_pop.size`.
    t_start:
        The starting time in the considered time window (included)
    t_end:
        The ending time in the considered time window (included) (why?)
    ax:
        a Matplotlib axis
    """
    spike_t = spike_t[spike_ids == idx * factor]
    spike_t = spike_t[spike_t >= t_start]
    spike_t = spike_t[spike_t <= t_start+t_end]
    # st = st[::10]
    spike_t = np.unique(spike_t) # sometimes spikes are duplicated (how come?)
    spike_t = np.reshape(spike_t, (1, -1))
    x = np.vstack((spike_t, spike_t))
    y = np.ones(x.shape)
    y[0, :] = -70.0
    y[1, :] = 20.0
    ax.plot(x, y, 'k', linewidth=0.2)

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

    data, protocol = parse_data(param, exp_name, to_read)

    plt.rc('font', size=8)
    ra = data["or_ra"]

    # select a timestep
    print(param.keys())

    dt = param["simulations"]["simulation"]["dt"] * param["simulations"]["simulation"]["n_timesteps_to_pull_var"]
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
        filename = f"{param['simulations']['simulation']['output_path']}/{exp_name}/{exp_name}_raw_spikes_{i}_{name}.png"
        print(f"saving to {filename}")
        plt.savefig(filename, dpi=1000)
        print(f"saved")


def select_window(time_data: np.ndarray, protocol: Protocol, i):
    # Find a window that is contained by a protocol event in O(log N)
    t_start = protocol.events[i]["t_start"]
    t_end_event = protocol.events[i]["t_end"]
    t_end_rest = protocol.events[i]["t_end"] + protocol.resting_duration

    left = np.searchsorted(time_data, t_start)
    right_event = np.searchsorted(time_data, t_end_event, 'right')
    right_rest = np.searchsorted(time_data, t_end_rest, 'right')

    return left, right_event, right_rest

def plot_heatmap(param, exp_name, **kwargs):
    data, protocol = parse_data(
        param,
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
        param,
        exp_name, {
            "orn": ["spikes"]#, "pn": ["spikes"], "ln": ["spikes"]
        }
    )

    plt.rc("font", size=8)
    plt.rcParams['text.usetex'] = True
    n_or = param["neuron_populations"]["or"]["n"]

    if not isinstance(protocol, FirstProtocol):
        raise ValueError("This plot requires FirstProtocol")
    for pop in ["orn"]:
        concentrations = {"iaa": [0], "geo": [0]}
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

def plot_mono(param, exp_name, **kwargs):
    # Abridged from Prof. Nowotny's notebook:
    # Monotonicity is the difference between the maximal value of the time-averaged
    # SDF response and the value at the maximal concentration.
    # - If mono is high, the meanSDF concentratin curve has a local maximum at a lower concentration.
    # - If mono is zero, then there is a high correlation between meanSDF and

    data, protocol = parse_data(
        param,
        exp_name, {
            "orn": ["spikes"], "pn": ["spikes"], "ln": ["spikes"]
        }
    )

    if not isinstance(protocol, FirstProtocol):
        raise ValueError("This plot requires ThirdProtocol")

    protocol = cast(FirstProtocol, protocol)

    pn_spike_t = data["pn_spikes"][:, 0]
    pn_spike_ids = data["pn_spikes"][:, 1]
    n_pop = param["neuron_populations"]["pn"]["n"]
    n_or = param["neuron_populations"]["or"]["n"]

    sigma_sdf = 100.0
    dt_sdf = 1.0


    # FIXME
    n_odors = 3
    n_conc = 3

    max_activation_per_app = np.array((n_odors, n_conc))
    avg_activation_per_app = np.array((n_odors, n_conc))
    max_glo_per_odor = np.array((n_odors,))

    for odor in range(n_odors):
        total_mean_sdf_per_odor = np.zeros(n_or)
        for conc in range(n_conc):
            cur_step = protocol.events[odor*3 + conc]
            #t_start = cur_step["t_start"]
            #t_end = cur_step["t_end"]
            left_active_idx, _, right_inactive_idx = select_window(pn_spike_t, protocol, i)
            spikes_t = pn_spike_t[left_active_idx:right_inactive_idx]
            spikes_ids = pn_spike_ids[left_active_idx:right_inactive_idx]

            sdf = make_sdf(spikes_t, spikes_ids, n_pop, n_pop, dt_sdf, sigma_sdf)
            factor = n_pop // n_or
            avg_glo_sdf = glo_avg(sdf, factor)
            avg_glo_sdf_onlyactive = avg_glo_sdf[3000:6000, :] # this is nasty
            mean_sdf_per_odor = np.mean(avg_glo_sdf_onlyactive, axis=0)
            max_sdf_per_odor = np.amax(avg_glo_sdf, axis=0)
            total_mean_sdf_per_odor += mean_sdf_per_odor

        '''
        max_glo_per_odor[i] = np.argmax(total_mean_sdf_per_odor)
        for conc in range(n_conc):
            max_activation_per_app[odor, conc] = max_sdf_per_odor
            avg_activation_per_app[odor, conc] = mean_sdf_per_odor
        '''


    max_mono = np.zeros(n_odors)
    mean_mono = np.zeros(n_odors)
    '''
    for i in range(n_odors):
        # ("max activation ever recorded for an odor" - "maximum activation at the maximum concentration")
        # ------------------------------------------------------------------------------------------------
        #                ("average of [maximum activation of each concentration] ")
        max_mono[i] = (np.amax(max_activation_per_app[i, :]) - max_activation_per_app[i, -1]) /
        mean_mono[i] = (np.amax(avg_activation_per_app[i, :]) - max_activation_per_app[i, -1]) /
    '''


if __name__ == "__main__":
    parser = get_argparse_template()
    plot_group = parser.add_argument_group("plot")
    plot_group.add_argument("plot", choices=['spikes', 'heatmap', 'sdf-over-c', 'mono'])
    plot_group.add_argument("--precision", help="For spikes plotting, set the desired resolution (in ms). Defaults to the simulation dt.")
    params = parse_cli(parser)
    name = params['simulations']['name']

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
