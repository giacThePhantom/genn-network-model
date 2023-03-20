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
from second_protocol import SecondProtocol

import numpy as np
import tables
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


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
    recorded_vars: dict
        a dictionary of variables that will be recorded
    param: dict
        configuration root
    """

    def _reset(self):
        self.recorded_vars = {}
        self.model.reinitialize()

    def __init__(
        self,
        sim_name: str,
        protocol: Protocol,
        param: dict
    ) -> None:
        self.sim_name = sim_name

        self.model = NeuronalNetwork(
            param['simulation']['name'],
            param['neuron_populations'],
            param['synapses'],
            param['simulation']['simulation']['dt'],
            optimizeCode=params['simulation']['simulation']['optimize_code'],
            generateEmptyStatePush=params['simulation']['simulation']['generate_empty_state_push']
        )

        self.recorded_vars = {}
        # data -> population -> var name -> array.
        # For each layer automatically create a default dict
        self._data = defaultdict(lambda: defaultdict(lambda: np.array([])))
        self.protocol = protocol
        self.param = param['simulation']['simulation']
        batch = self.local_var_batch = self.param['batch']
        dt = self.param['dt']
        self._output_table = None
        self.batch_size_timesteps = round(batch / dt)
        self._reset_population()
        self.track_vars()

    def _track_var(self, population: str, var_names: List[str], f: tables.File):
        group = f.create_group(f.root, population)
        for name in var_names:
            # pop = self.model.neuron_populations[population]
            genn_pop = self.model.connected_neurons[population]
            if name == "spikes":
                genn_pop.spike_recording_enabled = True
                target_cols = 2
            else:
                # TODO: implement CUDA event buffers for those variables.
                # For now all we can do is pulling them manually every n steps

                # Format: (v_1, v_2, ..., v_k, t) with k being the number of neurons.
                target_cols = genn_pop.size + 1
                self._data[population][name] = np.empty((self.batch_size_timesteps, genn_pop.size + 1))

            # For spikes, this is hopefully an upper bound. For vars, this is exact.
            expected_rows = self.protocol.simulation_time // self.param["dt"]
            print(expected_rows)
            self.recorded_vars.setdefault(population, []).append(name)
            f.create_earray(group, name, tables.Float64Atom(),
                            (0, target_cols), expectedrows=expected_rows)


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

        self.dirpath = Path(self.param["output_path"]) / self.sim_name
        self.logging_path = self.dirpath / "tracked_vars.h5"
        self.protocol_path = self.dirpath / "protocol.pickle"
        self.dirpath.mkdir(exist_ok=True)
        self._row_count = 0

        with self.protocol_path.open("wb") as f:
            pickle.dump(self.protocol, f)

        variables = [(key, self.param['tracked_variables'][key])
                     for key in self.param['tracked_variables']]

        self.filters = tables.Filters(complib='blosc:zstd', complevel=5)
        with tables.open_file(self.logging_path, 'w', filters=self.filters) as f:
            for var in variables:
                self._track_var(*var, f=f)

    def _reset_population(self):
        self._row_count = 0

        #for pop, var_dict in self._data.items():
        #    for var, values in var_dict.items():
        #        self._data[pop][var].fill(0)


    def _stream_output(self):
        logging.info(f"Saving to {self.dirpath}")

        # This makes sure the table is locked until the simulation ends (graciously or not).
        if self._output_table is None:
            self._output_table = tables.open_file(self.logging_path, 'a', self.filters)

        for pop, var_dict in tqdm(self._data.items()):
            for var, values in var_dict.items():
                # handle both spiking events and snapshots
                if len(values.shape) == 1:
                    values = values.reshape((1, -1))
                self._output_table.root[pop][var].append(values)

        self._reset_population()

    def _flush(self):
        self._output_table.close()
        self._output_table = None
        self._reset_population()

    def _add_to_var(self, pop, var, times, series):
        if var != "spikes":
            #print(pop, var, self._data[pop][var].shape)
            self._data[pop][var][self._row_count] = np.concatenate([times, series])
        else:
            self._data[pop][var] = np.column_stack([times, series])

    def update_target_pop(self, target_pop, current_events, events):
        for (i, event) in enumerate(current_events):
            if self.model.network.t >= event['t_start'] and not event['happened']:
                event['happened'] = True
                target_pop.vars["kp1cn_" +
                                str(event['channel'])].view[:] = event['binding_rates']
                target_pop.vars["kp2_" + str(event['channel'])
                                ].view[:] = event['activation_rates']
                self.model.network.push_state_to_device("or")

            elif self.model.network.t == event['t_end']:
                target_pop.vars["kp1cn_" + str(event['channel'])].view[:] = np.zeros(
                    np.shape(event['activation_rates']))

                if events[i]:
                    current_events[i] = events[i].pop(0)

    def _collect_spikes(self, poll_spike_readings):
        genn_model = self.model.network
        if not poll_spike_readings:
            genn_model.pull_recording_buffers_from_device()
        for pop_name in self.recorded_vars:
            if "spikes" not in self.recorded_vars[pop_name]:
                continue
            genn_pop = self.model.connected_neurons[pop_name]
            if not poll_spike_readings:
                spike_t = genn_pop.spike_recording_data[0]
                spike_id = genn_pop.spike_recording_data[1]
            else:
                genn_pop.pull_current_spikes_from_device()
                spike_count = genn_pop.spike_count[0][0]
                logging.debug(f"Detected {spike_count} spike events")
                if spike_count > 0:
                    # realistically, spike_count will hardly be bigger than 1
                    spike_t = genn_model.t * np.ones(spike_count)
                    spike_id = genn_pop.spikes[0][0][:spike_count]
                else:
                    spike_t = []
                    spike_id = []
            self._add_to_var(pop_name, "spikes", spike_t, spike_id)
    

    def _collect_vars(self):
        # Collect and save the variables during the simulation
        # TODO: split into _collect_spikes and _collect_vars

        genn_model = self.model.network
        for pop_name, pop_vars in self.recorded_vars.items():
            genn_pop = self.model.connected_neurons[pop_name]
            for var in pop_vars:
                if var == "spikes":
                    continue
                genn_pop.pull_var_from_device(var)
                logging.debug(f"{pop_name} -> {var}")
                logging.debug(genn_pop.vars[var].view)
                series = genn_pop.vars[var].view.T
                times = np.array([genn_model.t])

                self._add_to_var(pop_name, var, times, series)

        self._row_count += 1


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
        genn_model = self.model.network
        logging.info(
            f"Starting a simulation for the model {genn_model.model_name} that will run for {self.protocol.simulation_time} ms")

        if not genn_model._built:
            logging.info("Build and load")
            self.batch_size_timesteps
            self.model.build_and_load(self.batch_size_timesteps)
            logging.info("Done")
        else:
            logging.info("Reinitializing")
            self.model.reinitialise()

        events = self.protocol.get_events_for_channel()
        current_events = []
        for i in events:
            if i:
                current_events.append(i.pop(0))

        target_pop = self.model.connected_neurons['or']

        # Kickstart the simulation
        batch_timesteps = self.batch_size_timesteps
        total_timesteps = round(self.protocol.simulation_time / genn_model.dT)

        with logging_redirect_tqdm():
            with tqdm(total=total_timesteps) as pbar:
                while genn_model.t < self.protocol.simulation_time:
                    logging.debug(f"Time: {genn_model.t}")
                    genn_model.step_time()
                    self.update_target_pop(target_pop, current_events, events)
                    self._collect_vars()

                    if genn_model.timestep % batch_timesteps == 0:
                        self._collect_spikes(poll_spike_readings)
                        if save:
                            self._stream_output()
                    pbar.update()
        self._flush()

# TODO yeet out
class TestFirstProtocol(FirstProtocol):
    def events_generation(self, _):
        """Creates the event for the protocol and saves them in a private field
        Parameters
        ----------
        num_concentration_increases : int
            The number of times the concentration is increased by a dilution factor
        """
        res = []
        t = self.resting_duration
        for (i, odor) in enumerate(self.odors):
            if i >= 3:
                break
            for c_exp in range(15, 18):
                res.append(self._event_generation(t, odor, c_exp))
                t = res[-1]['t_end'] + self.resting_duration
        self.events = res

def pick_protocol(params):
    # Pick the correct protocol for the experiment
    protocol_data = params["protocols"]
    match params["simulation"]["simulation"]["experiment_name"]:
        case "experiment1":
            protocol = FirstProtocol(protocol_data["experiment1"])
        case "experiment2":
            protocol = SecondProtocol(protocol_data["experiment2"])
        case "testexperiment":
            protocol = TestFirstProtocol(protocol_data["experiment1"])

    protocol.add_inhibitory_conductance(
        params['synapses']['ln_pn'], params['neuron_populations']['ln']['n'], params['neuron_populations']['pn']['n'])
    protocol.add_inhibitory_conductance(
        params['synapses']['ln_ln'], params['neuron_populations']['ln']['n'], params['neuron_populations']['ln']['n'])

    return protocol


if __name__ == "__main__":
    params = parse_cli()
    protocol = pick_protocol(params)
    print(protocol.simulation_time)

    sim_params = params['simulation']
    sim = Simulator(sim_params['name'], protocol,
                    params)

    sim.run(
        batch=sim_params['simulation']['batch'],
        poll_spike_readings=sim_params['simulation']['poll_spike_readings'],
        save=True
    )