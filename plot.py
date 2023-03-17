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
from matplotlib import pyplot as plt

from reading_parameters import get_parameters

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
    print(sdf.shape)
    nglo= sdf.shape[1]//n
    gsdf= np.zeros((sdf.shape[0],nglo))
    for i in range(nglo):
        gsdf[:,i]= np.mean(sdf[:,n*i:n*(i+1)],axis=1)
    return gsdf

def parse_data(exp_name, vars_to_read: Dict[str, List[str]]) -> Tuple[Dict[str, np.ndarray], Protocol]:
    dirpath = Path("outputs") / exp_name
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
    st = np.reshape(st, (1, -1))
    x = np.vstack((st, st))
    y = np.ones(x.shape)
    y[0, :] = -70.0
    y[1, :] = 20.0
    ax.plot(x, y, 'k', lw=0.8)


def plot_spikes(param, exp_name):
    data, protocol = parse_data(exp_name, {
        "or": ["ra"],
        "orn": ["V", "spikes"],
        "pn": ["V", "spikes"],
        "ln": ["V", "spikes"]
    })

    #for event in protocol.events:
    #    pprint(event)

    orn_spikes = data["orn_spikes"]
    orn_spikes_t = orn_spikes[:, 0]
    orn_spikes_ids = orn_spikes[:, 1]

    pn_spikes = data["pn_spikes"]
    pn_spikes_t = pn_spikes[:, 0]
    pn_spikes_ids = pn_spikes[:, 1]

    ln_spikes = data["ln_spikes"]
    ln_spikes_t = ln_spikes[:, 0]
    ln_spikes_ids = ln_spikes[:, 1]

    ra = data["or_ra"]

    # select a timestep
    # dt = protocol.param["simulation"]["dt"]
    dt = 0.2
    ra_times = ra[:, 0]

    for i in range(3):
        if isinstance(protocol, FirstProtocol):
            cur_step = protocol.events[i]
        else:
            cur_step = protocol.events[2*i]
        #trial_time = end_step["t_end"] - cur_step["t_start"]
        t_off = cur_step["t_start"]
        t_end = cur_step["t_end"] + protocol.resting_duration
        t = np.arange(t_off, t_end, dt)
        
        # n_points = len(t)

        #i_off = int((i * trial_time + offset)/dt)
        #t_off = i_off*dt
        figure, ax = plt.subplots(4, sharex=True)
        figure.title = f"Spike activation at iteration {i}"
        # Pick the strongest glomerulus at the current experiment

        # First, pick the right time
        start_ra = np.searchsorted(ra_times, t_off)
        end_ra = np.searchsorted(ra_times, t_end, 'right')
        ra_my_times = ra_times[start_ra:end_ra]
        ra_vals = ra[start_ra:end_ra, 1:]
        print(ra_my_times)
        # Then, sum the overall activation per neuron
        ra_sum = np.sum(ra_vals, axis=0)
        ra_most_active = np.argmax(ra_sum)

        # plot the most active neuron over time
        ax[0].set_title(f"OR best ra activation rate (of neuron {ra_most_active})")
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].plot(t, ra_vals[:,ra_most_active], 'k', linewidth=0.5)
        print(t[0], t[-1])
        ax[0].set_xlim([t[0], t[-1]])
        ax[0].set_ylim([-0.025, 1.5*np.amax(ra_vals[:, ra_most_active])])

        ax[0].add_patch(
            Rectangle((100, -0.025), 3000, 0.2*np.amax(ra_vals[:, ra_most_active]),
                      edgecolor='grey',
                      facecolor='grey',
                      fill=True)
        )
        '''

        # same, with ORN voltage based on ra output
        vorn = data["orn_V"][start_ra:start_ra+n_points, 1:]
        or_n = param["neuron_populations"]["or"]["n"]
        orn_n = param["neuron_populations"]["orn"]["n"]
        factor = orn_n // or_n
        print(ra_most_active)
        ax[1].set_title(f"ORN V for neuron {ra_most_active*factor}")
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].plot(t, vorn[:,ra_most_active*factor], 'k', linewidth=0.5)
        subplot_spike_pane(orn_spikes_t, orn_spikes_ids, ra_most_active, factor, t_off, t_end, ax[1])
        ax[1].set_ylim([-90, 40])
        ax[1].set_ylabel("mV")
        
        vpn = data["pn_V"][start_ra:start_ra+n_points, 1:]
        pn_n = param["neuron_populations"]["pn"]["n"]
        factor = pn_n // or_n
        ax[2].set_title(f"PN V for neuron {ra_most_active*factor}")
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['top'].set_visible(False)
        ax[2].plot(t, vpn[:,ra_most_active*factor], 'k', linewidth=0.5)
        subplot_spike_pane(pn_spikes_t, pn_spikes_ids, ra_most_active, factor, t_off, t_end, ax[2] )
        ax[2].set_ylim([-90, 40])
        ax[2].set_ylabel("mV")

        vln = data["ln_V"][start_ra:start_ra+n_points, 1:]
        ln_n = param["neuron_populations"]["ln"]["n"]
        factor = ln_n // or_n
        ax[3].set_title(f"LN V for neuron {ra_most_active*factor}")
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['top'].set_visible(False)
        ax[3].plot(t, vln[:,ra_most_active*factor], 'k', linewidth=0.5)
        subplot_spike_pane(ln_spikes_t, ln_spikes_t, ra_most_active, factor, t_off, t_end, ax[2] )
        ax[3].set_ylim([-90, 40])
        ax[3].set_ylabel("mV")

        '''
        plt.show()



def plot_heatmap(param, exp_name):
    data, protocol = parse_data(exp_name, {"or": ["ra"], "orn": ["spikes"]})
    pn_spikes = data["orn_spikes"]

    n_or = param["neuron_populations"]["or"]["n"]
    n_pn = param["neuron_populations"]["orn"]["n"]
    or_ra = data["or_ra"]
    pn_spikes_t = pn_spikes[:, 0]
    pn_spikes_ids = pn_spikes[:, 1]
    sdfs = []
    glo_avg_sdfs = []


    # select some interesting experiments
    # in the first experiment, for example, consider
    # 

    for i in range(0, len(protocol.events), 4):
        od_active = protocol.events[i]
        od_inactive = protocol.events[i+2]
        od_active_tstart = od_active["t_start"]
        od_inactive_tend = od_inactive["t_end"]
        print(od_active_tstart, od_inactive_tend)

        left_spikes_active_idx = np.searchsorted(pn_spikes_t, od_active_tstart)
        #right_spikes_active_idx = np.searchsorted(pn_spikes_t, od_active_tend, 'right')
        #left_spikes_inactive_idx = np.searchsorted(pn_spikes_t, od_active_tstart)

        #active_spikes_t = pn_spikes_t[left_spikes_active_idx:right_spikes_active_idx]
        #active_spikes_ids = pn_spikes_ids[left_spikes_active_idx:right_spikes_active_idx].astype(np.uint32)


        #left_spikes_inactive_idx = np.searchsorted(pn_spikes_t, od_inactive_tstart)
        right_spikes_inactive_idx = np.searchsorted(pn_spikes_t, od_inactive_tend, 'right')

        
        spikes_t = pn_spikes_t[left_spikes_active_idx:right_spikes_inactive_idx]
        spikes_ids = pn_spikes_ids[left_spikes_active_idx:right_spikes_inactive_idx]

        if len(spikes_t) == 0:
            break
        
        #inactive_spikes_t = pn_spikes_t[left_spikes_inactive_idx:right_spikes_inactive_idx]
        #inactive_spikes_ids = pn_spikes_ids[left_spikes_inactive_idx:right_spikes_inactive_idx].astype(np.uint32)

        sigma_sdf = 100.0
        dt_sdf = 1.0

        print(spikes_ids)
        print(n_pn)

        sdfs.append(make_sdf(spikes_t,
                    spikes_ids, n_pn, dt_sdf, sigma_sdf))
        
        # Assume each glomerulus is made of 5 neurons (?)
        glo_avg_sdfs.append(glo_avg(sdfs[-1], 5))

        print(sdfs[-1].T)

        plt.imshow(sdfs[-1].T, cmap='hot')
        plt.title("Glomeruli spike activation in PN")
        plt.xlabel("Time")
        plt.ylabel("Neuron group")
        
        '''
        or_ra_selected = or_ra[0:9000*5, 1:]
        xticks = np.arange(0, 9000, 500)
        plt.imshow(or_ra_selected.T, cmap='hot')#, extent=[0, 9000, 0, n_or])
        plt.title("Glomeruli spike activation in PN")
        plt.xlabel("Time")
        plt.ylabel("RA activation")
        
        '''
        plt.show()
        break


if __name__ == "__main__":
    import sys
    plot_spikes(get_parameters(sys.argv[1]), sys.argv[2])
    #plot_heatmap(get_parameters(sys.argv[1]), sys.argv[2])