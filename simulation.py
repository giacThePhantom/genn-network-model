from collections import deque
from pathlib import Path
import pickle
from typing import List, Tuple
import logging

from network import NeuronalNetwork
from neuron import NeuronPopulation
from protocols import Protocol, ProtocolStep, exp1_protocol
from odors import Odor

import numpy as np

class Simulator:
    """
    A Simulator automatically launches a network with a given protocol
    and tracks the required populations.

    Attributes
    ----------
    sim_name: str
        The name of the simulation
    model: NeuronalNetwork
        a neuronal network to simulate and track
    
    protocol: a protocol to simulate (TODO)
    """

    def _reset(self):
        self.recorded_vars = {}
        self.model.reinitialize()

    def __init__(
        self,
        sim_name: str,
        model: NeuronalNetwork,
        protocol: Protocol,
        odor: np.ndarray, # FIXME
        hill_n: np.ndarray
    ) -> None:
        self.sim_name = sim_name
        self.model = model
        self.protocol = protocol
        self.odor = odor
        self.hill_n = hill_n
        self.recorded_vars = {}

    def _track_var(self, population: str, var_names: List[str]):
        for name in var_names:
            pop = self.model.neuron_populations[population]
            genn_pop = self.model.connected_neurons[population]
            if name == "spikes":
                print("recording...")
                genn_pop.spike_recording_enabled = True
            else:
                # TODO: implement CUDA event buffers for those variables.
                # For now all we can do is pulling them manually every n steps
                pass
            self.recorded_vars.setdefault(population, []).append(name)

            pop.recorded_outputs[name] = np.array([])

    def track_vars(self, variables: List[Tuple[str, List[str]]]):
        """
        Track a list of variables per population

        Parameters
        ----------
        variables: list
            keeps track of which variables to pull from the neurons.
            Each element is in the form (pop_name, [var1, var2, ...]) where
            var1, ... are variables defined in the specific neuron code used by that population.
            An extra variable, "spikes", enables spike counting (ie. when V >= some threshold)
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

    def save_output(self):
        path = Path(self.sim_name + ".pickle")
        logging.info(f"Saving to {path}")
        to_log = {}
        for pop_name, pop in self.model.neuron_populations.items():
            to_log[pop_name] = pop.recorded_outputs

        with path.open("wb") as f:
            pickle.dump(to_log, f)

    def _add_to_var(self, pop, var, times, series):
        if len(pop.recorded_outputs[var]) > 0:
            cur_buf_size = pop.recorded_outputs[var].shape[1]
        else:
            cur_buf_size = 0
        pop.recorded_outputs[var].resize(2, cur_buf_size + len(times))
        pop.recorded_outputs[var][0, cur_buf_size:] = times
        pop.recorded_outputs[var][1, cur_buf_size:] = series

    def run(self, howlong: float, batch=1.0, poll_spike_readings=False):
        """
        Run a simulation. The user is advised to call `track_vars` first
        to register which variables to log during the simulation
        
        Parameters
        ----------

        howlong: float
            for how long we want the simulation to run.
        batch: float
            how often (in ms) to pull data from the GPU. Must be a multiple of dt.
            We recommend not keeping this value too low or the GPU may stall.
        poll_spike_readings: bool
            if False (default), use the internal SpikeRecorder class to record spike events.
            This is much faster than polling the internal state, but is limited to the internal implementation.
            Otherwise, use the (old) spike event polling method. This means almost all events will be lost between
            readings, however it provides useful "snapshot" views for debugging.
        """
        protocols = deque(self.protocol)
        model = self.model.network
        logging.info(f"Starting a simulation for the model {model.model_name} that will run for {howlong} ms")

        if not model._built:
            self.model.build_and_load(int(batch / model.dT))
        else:
            self.model.reinitialise()
        
        # FIXME
        while model.t < howlong:
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

            model.step_time()
        
            if model.t > 0 and np.isclose(np.fmod(model.t, batch), 0.0):
                print(f"Time: {model.t}")
                for pop_name, pop_vars in self.recorded_vars.items():
                    pop = self.model.neuron_populations[pop_name]
                    genn_pop = self.model.connected_neurons[pop_name]
                    for var in pop_vars:
                        if var == "spikes":
                            if not poll_spike_readings:
                                model.pull_recording_buffers_from_device()
                                spike_t = genn_pop.spike_recording_data[0]
                                spike_id = genn_pop.spike_recording_data[1]
                            else:
                                genn_pop.pull_current_spikes_from_device()
                                spike_count = genn_pop.spike_count[0][0]
                                logging.debug(f"Detected {spike_count} spike events")
                                if spike_count > 0:
                                    # realistically, spike_count will hardly be bigger than 1
                                    spike_t = model.t * np.ones(spike_count)
                                    spike_id = genn_pop.spikes[0][0][:spike_count]
                                else:
                                    spike_t = []
                                
                            if len(spike_t) == 0:
                                continue

                            self._add_to_var(pop, var, spike_t, spike_id)
                            
                            logging.debug(f"pop: {pop_name}, spike_t: {spike_t}, spike_id: {spike_id}")
                        else:
                            genn_pop.pull_var_from_device(var)
                            logging.debug(f"{pop_name} -> {var}")
                            logging.debug(genn_pop.vars[var].view)
                            series = genn_pop.vars[var].view
                            times = model.t * np.ones_like(series)

                            self._add_to_var(pop, var, times, series)
        
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

    logging.basicConfig(level=logging.DEBUG)
    sim = Simulator("test_sim", model, protocol, odor_matrix, hill_n)
    sim.track_vars([("or", ["kp1cn_0"]), ("ln", ["spikes"]), ("pn", ["spikes"])])
    sim.run(5000.0, batch=1000.0, poll_spike_readings=False)
    sim.save_output()

    # model.build_and_load() # no longer necessary