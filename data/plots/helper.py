import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, TYPE_CHECKING

import tables

if TYPE_CHECKING:
    from protocol import Protocol
    from first_protocol import FirstProtocol
    from second_protocol import SecondProtocol
    from third_protocol import ThirdProtocol
    from test_protocol import TestFirstProtocol

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
 