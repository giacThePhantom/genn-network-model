from collections import deque
from typing import List, Tuple

from network import NeuronalNetwork
from protocols import Protocol, ProtocolStep, exp1_protocol
from odors import Odor
from first_protocol import FirstProtocol

import numpy as np

class Simulator:
    """
    A Simulator automatically launches a network with a given protocol
    and tracks the required populations.

    Attributes
    ----------
    model: NeuronalNetwork
        a neuronal network to simulate and track

    protocol: a protocol to simulate
    """

    def __init__(
        self,
        model: NeuronalNetwork,
    ) -> None:
        self.model = model

    def _track_var(self, population: str, var_name: str):
        pass  # TODO

    def track_vars(self, variables: List[Tuple[str, str]]):
        """
        Track a list of variables
        """
        for var in variables:
            self._track_var(*var)


    def _clear(self):
        for i in range(3):
            self._reset_population(i)
        self.model.network.push_state_to_device("or")

    def run(self, until: float, batch=1.0):
        """
        Run a simulation. The user is advised to call `track_vars` first
        to register which variables to log during the simulation

        Parameters
        ----------

        until: float
            for how long we want the simulation to run.
        batch: int
            how often (in ms) to pull data from the GPU. Must be a multiple of dt.
            We recommend not keeping this value too low or the GPU may stall.

        """
        model = self.model.network

        if not model._built:
            self.model.build_and_load()

        # FIXME
        while model.t < until:
            print(f"Time: {model.t}")
            model.step_time()


if __name__ == "__main__":
    import sys
    from reading_parameters import get_parameters
    params = get_parameters(sys.argv[1])
    first_protocol = FirstProtocol(params['protocols']['experiment1'])
    first_protocol.events_generation(1)
    first_protocol.generate_or_param(params['neuron_populations']['or'])
    model = NeuronalNetwork(
        "Test", params['neuron_populations'], params['synapses'])

    sim = Simulator(model)
    sim.run(500.0)

    # model.build_and_load() # no longer necessary
