from collections import deque
from typing import List, Tuple

from network import NeuronalNetwork
from protocols import Protocol, ProtocolStep, exp1_protocol
from odors import Odor

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
        protocol: Protocol,
        odor: np.ndarray, # FIXME
        hill_n: np.ndarray
    ) -> None:
        self.model = model
        self.protocol = protocol
        self.odor = odor
        self.hill_n = hill_n

    def _track_var(self, population: str, var_name: str):
        pass  # TODO

    def track_vars(self, variables: List[Tuple[str, str]]):
        """
        Track a list of variables
        """
        for var in variables:
            self._track_var(*var)

    def _reset_population(self, odor, concentration=0.0):
        # for each slot/odor
        ors = self.model.connected_neurons["or"]
        odors = self.odor
        od = odors[odor, :, :]
        kp1cn = np.power(od[:, 0] * concentration, self.hill_n)
        kp2 = od[:, 1]

        km1 = 0.025
        kp2 = od[:, 1]
        km2 = 0.025

        slot = str(odor)

        vname = "kp1cn_"+slot
        ors.vars[vname].view[:] = kp1cn
        vname = "km1_"+slot
        ors.vars[vname].view[:] = km1
        vname = "kp2_"+slot
        ors.vars[vname].view[:] = kp2
        vname = "km2_"+slot
        ors.vars[vname].view[:] = km2

    def _apply_step(self, step: ProtocolStep):
        # FIXME remove me after protocol class is merged

        ors = self.model.connected_neurons["or"]
        slot = step.odor # step is ALWAYS an odor slot(?)

        self._reset_population(slot, step.concentration)

        self.model.push_state_to_device("or")

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
        protocols = deque(self.protocol)
        model = self.model.network

        if not model._built:
            self.model.build_and_load()

        # FIXME
        while model.t < until:
            while len(protocols) and protocols[0].t_end >= model.t:
                protocols.popleft()
            
            for step in protocols:
                if step.t_start > self.model.t:
                    # FIXME: this actually doesn't clear *other* odors.
                    # But we don't care for now
                    break # we're in the future
                self._apply_step(step)
            else: # no step was found
                self._clear()

            print(f"Time: {model.t}")
            model.step_time()


if __name__ == "__main__":
    import sys
    from reading_parameters import get_parameters
    params = get_parameters(sys.argv[1])
    model = NeuronalNetwork(
        "Test", params['neuron_populations'], params['synapses'])
    temp = Odor(params['protocols']['experiment1']
                ['odors']['default'], 'iaa', 160, False)
    temp1 = Odor(params['protocols']['experiment1']
                ['odors']['default'], 'geo', 160, False)
    temp2 = Odor(params['protocols']['experiment1']
                ['odors']['default'], 'default', 160, False)

    temp.shuffle_binding_rates()
    temp1.shuffle_binding_rates()
    temp2.shuffle_binding_rates()

    odor_matrix = np.array(
        [temp.get_cuda_rates(), temp1.get_cuda_rates(), temp2.get_cuda_rates()])
    print(odor_matrix.shape)
    protocol = exp1_protocol()
    hill_n = np.random.uniform(0.95, 1.5, 160)

    sim = Simulator(model, protocol, odor_matrix, hill_n)
    sim.run(500.0)

    # model.build_and_load() # no longer necessary