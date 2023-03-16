"""
Make plots. TODO cleanup
"""

import math
from pathlib import Path
import pickle
from typing import Dict, List, Tuple

from matplotlib.patches import Rectangle
import numpy as np
import tables

from simulation import TestFirstProtocol
from protocol import Protocol
from matplotlib import pyplot as plt

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
    

def plot_spikes(exp_name):
    data, protocol = parse_data(exp_name, {"or": ["ra"], "or": ["V"], "pn": ["V"], "ln": ["spikes"]})
    
    pn_spikes = data["pn"]["spikes"]
    pn_spikes_t = pn_spikes[:, 0]
    pn_spikes_ids = pn_spikes[:, 1]

    ra = data["or"]["ra"]

    # select a timestep
    trial_time = 12000.0 # FIXME
    offset = 2900.0
    dt = protocol.param["simulation"]["dt"]
    t = np.arange(0, 3500, dt)
    ra_times = ra[:, 0]
    n_points = len(t)
    for i in range(4):
        i_off = int((i * trial_time + offset)/dt)
        t_off = i_off*dt
        figure, ax = plt.subplots(4, sharex=True)
        # Pick the strongest glomerulus at the current experiment

        # First, pick the right time
        start_ra = np.searchsorted(ra_times, t_off)
        ra_vals = ra[start_ra:start_ra+n_points, 1:]
        # Then, sum the overall activation per neuron
        ra_sum = np.sum(ra_vals, axis=0)
        ra_most_active = np.argmax(ra_sum)

        # plot the most active neuron over time
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].plot(t, ra_vals[:,ra_most_active], 'k', linewidth=0.5)
        ax[0].set_xlim([0, t[-1]])        
        ax[0].set_ylim([-0.025, 1.5*np.amax(ra[:, ra_most_active])])

        ax[0].add_patch(
            Rectangle((100, -0.025), 3000, 0.2*np.amax(ra_vals[:, ra_most_active]),
                      edgecolor='grey',
                      facecolor='grey',
                      fill=True)
        )

        # same, with ORN voltage based on ra output
        vorn = data["orn"]["V"][start_ra:start_ra+n_points, 1:]
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].plot(t, vorn[:,ra_most_active], 'k', linewidth=0.5)
        ax[1].set_ylim([-90, 40]) # wait what?
        
        # TODO




def plot_heatmap(exp_name):
    pn_spikes, protocol = parse_data(exp_name)
    pn_spikes_t = pn_spikes[:, 0]
    pn_spikes_ids = pn_spikes[:, 1]
    sdfs = []
    glo_avg_sdfs = []

    N = 800

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

        sdfs.append(make_sdf(spikes_t,
                    spikes_ids, N, dt_sdf, sigma_sdf))
        
        # Assume each glomerulus is made of 5 neurons (?)
        glo_avg_sdfs.append(glo_avg(sdfs[-1], 5))

        # TODO


if __name__ == "__main__":
    import sys
    plot_heatmap(sys.argv[1])