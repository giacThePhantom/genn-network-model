"""
Make plots. TODO cleanup
"""

import math
from pathlib import Path
import pickle
import numpy as np

from second_protocol import SecondProtocol
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

    to = spike_times[0]
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


if __name__ == "__main__":
    dirpath = Path("test_sim")
    with (dirpath / "tracked_data.pickle").open("rb") as f:
        data = pickle.load(f)
    with (dirpath / "protocol.pickle").open("rb") as f:
        protocol: SecondProtocol = pickle.load(f)

    pn_spikes = data["pn"]["spikes"]
    pn_spikes_t = pn_spikes[0]
    pn_spikes_ids = pn_spikes[1]
    print(pn_spikes_t)

    # for now, let us look at IAA and Geosmin
    selected_odors = [0, 1]


    # FIXME all of this is a horrible mess.
    trial_time = 6000.0
    #concentrations = 25 # TODO
    #batch_t = concentrations * trial_time
    concentrations = 1
    batch_t = trial_time * 1 # TODO take concentrations from the protocol once we fix it
    #total_time = pn_spikes_t[-1]
    last_useful_idx = np.searchsorted(pn_spikes_t, batch_t)
    #print(last_useful_idx, pn_spikes_t[last_useful_idx])
    total_time = pn_spikes_t[last_useful_idx]

    odors_in_experiment = total_time // batch_t
    
    # Batch spikes by their belonging batch.
    # technically this is O(NlogN) but I don't care

    sdfs = []
    glo_avg_sdfs = []

    N = 800 # TODO: why???

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
        left = spikes_t[0]# - 3*sigma_sdf
        right = spikes_t[-1]# + 3*sigma_sdf
        
        sdfs.append(make_sdf(spikes_t,
                    spikes_ids, N, left, right, dt_sdf, sigma_sdf))
        
        # Assume each glomerulus is made of 5 neurons (?)
        glo_avg_sdfs.append(glo_avg(sdfs[-1], 5))

        #plt.imshow(sdfs[0].T)
        plt.imshow(glo_avg_sdfs[0].T, cmap='hot')
        plt.show()