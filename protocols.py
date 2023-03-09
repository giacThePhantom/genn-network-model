"""
A "protocol" is a set of operations that alter the intensity of an odour to the glomeruli for a specified amount of time.

A single operation is in the format:

    "t_start": when to start applying the odour (ms, inclusive)
    "t_end": when to stop applying the odour (ms, exclusive)
    "odor": which odour to apply (integer starting from 0)
    "concentration": the odour concentration (0 to disable)

For example:

{
    "t_start": 100,
    "t_end": 800,
    "odour": 2
    "concentration": 1e-5
}

After t_end, unless other steps are provided, the concentration will be reset to 0.

To apply multiple odours at the same time one can put two steps that overlap
(they don't need to exactly coincide). Eg.

[
    {
        "t_start": 100,
        "t_end": 800,
        "odor": 1"
        "concentration": 1e5
    },
    {
        "t_start": 150,
        "t_end": 700,
        "odor": 2
    }
]

This representation is unfortunately not very CUDA-friendly, which is why we provide
`convert_protocol_to_cuda`.
"""

from asyncio import protocols
from collections import deque
from copy import copy
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ProtocolStep:
    t_start: float
    t_end: float
    odor: int
    concentration: float

Protocol = List[ProtocolStep]

def convert_protocol_to_cuda(protocol: Protocol, odors, hill_n, row_major=True) -> np.ndarray:
    """
    Convert a protocol into a cuda-friendly matrix.
    The format is a 4 x N format, with the number of columns being
    the number of steps. This can be seen as the stacking of four horizontal arrays:
    - odor
    - t_start
    - t_end
    - concentration
    - kpcm (used in the differential equations governing the OR neuron)

    Parameters
    ----------
    protocol: List[ProtocolStep]
        the protocol to use. All steps must be sorted by time.
    odors: np.ndarray
        the odor matrix to use.
    hill_n: np.ndarray
        the exponents to use for each receptor
    row_major: bool
        if true (default) make the vectors horizontal.
    
    Returns
    -------

    A CUDA-friendly numpy array that can be passed to OR's kernel code
    """

    # Hacky way to do this
    _, n_glo_size, __ = odors.shape
    data = np.zeros((len(protocol), 4 + n_glo_size))

    for i, step in enumerate(protocol):
        odor = step.odor
        kp1cn = np.squeeze(np.power(odors[odor, :, 0]*step.concentration, hill_n))
        data[i] = np.hstack((
            np.array([np.float32(step.odor), step.t_start, step.t_end, step.concentration], dtype=np.float32),
            kp1cn))
    
    if row_major:
        return data.T

def exp1_protocol(n_odors = 3, n_conc=25):
    """
    For every odor, for the same trial, apply the odor for 3 seconds.
    Slightly increase the concentration per trial. Apply a 3-second gap between trials.
    """
    protocol = []
    base = np.power(10, 1/4) # every 4 steps we have a decuplication
    t_start = 0
    for odor in range(n_odors):
        for concentration in range(n_conc):
            protocol.append(
                ProtocolStep(
                    t_start,
                    t_start + 3000.0,
                    odor,
                    1e-7*np.power(base, concentration)
                )
            )
            protocol.append(
                ProtocolStep(
                    t_start + 3000.0,
                    t_start + 6000.0,
                    odor,
                    0.0
                )
            )
            t_start += 6000 # immediately apply the next odor
    
    return protocol

def exp2_protocol(n_conc=25, odor1=0, odor2=1):
    """
    Apply both the first and second odor at the same time for 3 seconds, but with
    different concentrations. Exaustively try all N^2 pairs. Make a 3-second break
    after exposure.
    """
    protocol = []
    base = np.power(10, 1/4) # every 4 steps we have a decuplication
    t_start = 0.0
    for c1 in range(n_conc):
        for c2 in range(n_conc):
            protocol.append(
                ProtocolStep(
                    t_start,
                    t_start + 3000.0,
                    odor1,
                    1e-7*np.power(base, c1),
                )
            )
            protocol.append(
                ProtocolStep(
                    t_start,
                    t_start + 3000.0,
                    odor2,
                    1e-7*np.power(base, c2),
                )
            )
            protocol.append(
                ProtocolStep(
                    t_start + 3000.0,
                    t_start + 6000.0,
                    odor1,
                    0.0
                )
            )
            protocol.append(
                ProtocolStep(
                    t_start + 3000.0,
                    t_start + 6000.0,
                    odor2,
                    0.0
                )
            )
            t_start += 6000
    return protocol

def exp3_protocol(odor1, odor2):
    # Similar to exp2, but with hand-picked concentrations
    
    protocol = []
    base = np.power(10, 1/4) # every 4 steps we have a decuplication
    t_start = 0.0
    for c1 in [0, 1e-3, 1e-1]:
        for c2 in [0, 1e-6, 1e-5, 1e-4, 1e-3]:
            protocol.append(
                ProtocolStep(
                    t_start,
                    t_start + 3000.0,
                    odor1,
                    c1
                )
            )
            if c2 != 0:
                protocol.append(
                    ProtocolStep(
                        t_start,
                        t_start + 3000.0,
                        odor2,
                        c2
                    )
                )
            protocol.append(
                ProtocolStep(
                    t_start + 3000.0,
                    t_start + 6000.0,
                    odor1,
                    0.0
                )
            )
            if c2 != 0:
                protocol.append(
                    ProtocolStep(
                        t_start + 3000.0,
                        t_start + 6000.0,
                        odor2,
                        0.0
                    )
                )
        t_start += 6000
    return protocol





if __name__ == "__main__":
    n_glo = 160 # TODO: add a neuron population and test here
    np.random.seed(0)
    hill_exp = np.random.uniform(0.95, 1.05, n_glo)
    odors = np.random.randn(100, n_glo, 2).astype(np.float32)


    print(convert_protocol_to_cuda(exp1_protocol(), odors, hill_exp))
    print(convert_protocol_to_cuda(exp2_protocol(), odors, hill_exp))
    print(convert_protocol_to_cuda(exp3_protocol(1, 2), odors, hill_exp))