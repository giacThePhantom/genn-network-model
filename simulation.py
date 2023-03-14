from collections import defaultdict, deque
from pathlib import Path
import pickle
from typing import List, Tuple
import logging

from network import NeuronalNetwork
from neuron import NeuronPopulation
from reading_parameters import parse_cli
from odors import Odor
from protocol import Protocol
from first_protocol import FirstProtocol

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
    protocol: Protocol
        the protocol for the simulation. This will be dumped.
    """

    def _reset(self):
        self.recorded_vars = {}
        self.model.reinitialize()

    def __init__(
        self,
        sim_name: str,
        model: NeuronalNetwork,
        protocol: Protocol,
        param : dict
    ) -> None:
        self.sim_name = sim_name
        self.model = model
        self.recorded_vars = {}
        # data -> population -> var name -> array.
        # For each layer automatically create a default dict
        self._data = defaultdict(lambda: defaultdict(lambda: np.array([])))
        self.protocol = protocol
        self.param = param
        self.track_vars()

    def _track_var(self, population: str, var_names: List[str]):
        for name in var_names:
            #pop = self.model.neuron_populations[population]
            genn_pop = self.model.connected_neurons[population]
            if name == "spikes":
                print("recording...")
                genn_pop.spike_recording_enabled = True
            else:
                # TODO: implement CUDA event buffers for those variables.
                # For now all we can do is pulling them manually every n steps
                
                # Format: (v_1, v_2, ..., v_k, t) with k being the number of neurons.
                self._data[population][name] = np.empty(genn_pop.size + 1)
            self.recorded_vars.setdefault(population, []).append(name)

    def track_vars(self):
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

        variables = [(key, self.param['tracked_variables'][key]) for key in self.param['tracked_variables']]
        for var in variables:
            self._track_var(*var)

    def _reset_population(self):
        self.model.clear_logs()

    def _stream_output(self):
        dirpath = Path("outputs") / self.sim_name
        logging.info(f"Saving to {dirpath}")
        dirpath.mkdir(exist_ok=True)

        protocol_path = dirpath / "protocol.pickle"
        if not protocol_path.exists():
            with protocol_path.open("wb") as f:
                pickle.dump(self.protocol, f)

        for pop, var_dict in self._data.items():
            for var, values in var_dict.items():
                logpath = dirpath / f"{pop}_{var}.csv"
                with logpath.open("a") as f:
                    np.savetxt(f, np.column_stack(values), delimiter=",", fmt='%.4e')

        self._reset_population()
    
    def _flush_output(self):
        dirpath = Path("outputs") / self.sim_name
        for file in dirpath.rglob("*"):
            file.unlink()
    
    def _add_to_var(self, pop, var, times, series):
        # We're saving a snapshot. On that case, squeeze the output into a single row
        if len(times) == 1:
            series = series.squeeze()
            self._data[pop][var] = np.concatenate([times, series])
        else:
            self._data[pop][var] = np.vstack([times, series])

    def update_target_pop(self, target_pop, current_events, events):
        for (i, event) in enumerate(current_events):
            if self.model.network.t >= event['t_start'] and not event['happened']:
                event['happened'] = True
                target_pop.vars["kp1cn_" + str(event['channel'])].view[:] = event['binding_rates']
                target_pop.vars["kp2_" + str(event['channel'])].view[:] = event['activation_rates']
                self.model.network.push_state_to_device("or")

            elif self.model.network.t == event['t_end']:
                target_pop.vars["kp1cn_" + str(event['channel'])].view[:] =np.zeros(np.shape(event['activation_rates']))

                if events[i]:
                    current_events[i] = events[i].pop(0)


    def run(self, batch=1.0, poll_spike_readings=False, save=True):
        """
        Run a simulation. The user is advised to call `track_vars` first
        to register which variables to log during the simulation

        Parameters
        ----------

        batch: float
            how often (in ms) to pull data from the GPU. Must be a multiple of dt.
            We recommend not keeping this value too low or the GPU may stall.
        poll_spike_readings: bool
            if False (default), use the internal SpikeRecorder class to record spike events.
            This is much faster than polling the internal state, but is limited to the internal implementation.
            Otherwise, use the (old) spike event polling method. This means almost all events will be lost between
            readings, however it provides useful "snapshot" views for debugging.
        """
        model = self.model.network
        logging.info(f"Starting a simulation for the model {model.model_name} that will run for {self.protocol.simulation_time} ms")
        self._flush_output()

        if not model._built:
            logging.info("Build and load")
            self.model.build_and_load(int(batch / model.dT))
            logging.info("Done")
        else:
            logging.info("Reinitializing")
            self.model.reinitialise()

        # FIXME
        events = self.protocol.get_events_for_channel()
        current_events = []
        for i in events:
            if i:
                current_events.append(i.pop(0))

        target_pop = self.model.connected_neurons['or']

        while model.t < self.protocol.simulation_time:
            logging.debug(f"Time: {model.t}")
            model.step_time()
            self.update_target_pop(target_pop, current_events, events)

            if model.t > 0 and np.isclose(np.fmod(model.t, batch), 0.0):
                print(f"Time: {model.t/1000}")
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

                            self._add_to_var(pop_name, var, spike_t, spike_id)

                            logging.debug(f"pop: {pop_name}, spike_t: {spike_t}, spike_id: {spike_id}")
                        else:
                            genn_pop.pull_var_from_device(var)
                            logging.debug(f"{pop_name} -> {var}")
                            logging.debug(genn_pop.vars[var].view)
                            series = genn_pop.vars[var].view.T
                            times = np.array([model.t])

                            self._add_to_var(pop_name, var, times, series)

                if save:
                    self._stream_output()

# TODO yeet out
class TestFirstProtocol(FirstProtocol):
    def __init__(self, param):
        param['concentration_increases'] = 3
        super().__init__(param)
    

if __name__ == "__main__":
    params = parse_cli()
    protocol = TestFirstProtocol(params['protocols']['experiment1'])
    protocol.add_inhibitory_conductance(params['synapses']['ln_pn'], params['neuron_populations']['ln']['n'], params['neuron_populations']['pn']['n'])
    protocol.add_inhibitory_conductance(params['synapses']['ln_ln'], params['neuron_populations']['ln']['n'], params['neuron_populations']['ln']['n'])


    model = NeuronalNetwork(
        params['simulation']['name'],
        params['neuron_populations'],
        params['synapses'],
        params['simulation']['simulation']['dt'],
        optimizeCode = params['simulation']['simulation']['optimize_code'],
        generateEmptyStatePush = params['simulation']['simulation']['generate_empty_state_push']
    )

    sim = Simulator("prot2_sim", model, protocol,
                    params['simulation']['simulation'])
    sim.run(batch=1000.0, poll_spike_readings=False, save=True)
