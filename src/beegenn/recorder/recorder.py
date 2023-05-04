import numpy as np
import logging
from pathlib import Path
import pickle
import tables
from typing import List
import pandas as pd
from copy import deepcopy

class Recorder:
    """
    An object that records and saves data during a simulation run

    Attributes
    ----------
    dirpath : pathlib.Path
        The directory where the output is stored
    logging_path : pathlib.Path
        The file where simulation data will be saved
    protocol_path : pathlib.Path
        Where the protocol will be saved
    filters : tables.Filters
        The compression method for the data
    n_timesteps_to_pull_var : int
        How often data is saved during the simulation
    n_points_in_batch : int
        Number of element in a batch
    batch : int
        How often data is saved to disk
    dt : float
        The duration of a timestep of the simulation
    simulation_time : float
        How long the simulation is
    recorded_vars : dict
        Which variables are going to be saved to disk

    Methods
    -------

    record(model, save) : None
        Saves the data to the disk
    dump_connectivity(model) : None
        Write the csv of the connectivity to disk
        NOT WORKING AT THE MOMENT
    """

    def __init__(self, output_path, sim_name, to_be_tracked, connected_neurons, batch, n_timesteps_to_pull_var,  dt, simulation_time):

        self.dirpath = Path(output_path) / sim_name
        self.logging_path = self.dirpath / "tracked_vars.h5"
        self.protocol_path = self.dirpath / "protocol.pickle"
        self.dirpath.mkdir(exist_ok=True)
        self.filters = tables.Filters(complib='blosc:zstd', complevel=5)

        self.n_timesteps_to_pull_var = n_timesteps_to_pull_var
        self.n_points_in_batch = round(batch / self.n_timesteps_to_pull_var)
        self.batch = batch
        self.dt = dt
        self.simulation_time = simulation_time

        self.recorded_vars = self._set_up_var_recording(to_be_tracked, connected_neurons)

        self._row_count = 0
        self._data = self._set_container_for_data_recording(self.recorded_vars, connected_neurons)
        self._output_table = None
        pass

    def dump_protocol(self, protocol):
        events = deepcopy(protocol.events)
        # for i in events:
        #     i.pop('binding_rates')
        #     i.pop('activation_rates')
        #     i.pop('happened')
        df = pd.DataFrame.from_dict(events, orient = 'columns')
        df.sort_values(by=['t_start', 't_end'])
        df.to_csv(str(self.dirpath / "events.csv"))
        with self.protocol_path.open('wb') as f:
            pickle.dump(protocol, f)

    def enable_spike_recording(self, model):
        for pop in self.recorded_vars:
            for var_name in self.recorded_vars[pop]:
                if var_name == 'spikes':
                    model.connected_neurons[pop].spike_recording_enabled = True

    def _set_container_for_data_recording(self, recorded_vars, connected_neurons):
        res = {}
        for pop in recorded_vars:
            res[pop] = {}
            for var_name in recorded_vars[pop]:
                if var_name != 'spikes':
                    res[pop][var_name] = np.empty((self.n_points_in_batch, connected_neurons[pop].size + 1))

        return res

    def _collect_vars(self, model):
        for pop in self.recorded_vars:
            genn_pop = model.connected_neurons[pop]
            for var_name in self.recorded_vars[pop]:
                if var_name != 'spikes':
                    genn_pop.pull_var_from_device(var_name)
                    logging.debug(f"{pop} -> {var_name}")
                    logging.debug(genn_pop.vars[var_name].view)
                    series = genn_pop.vars[var_name].view.T
                    times = np.array([model.network.t])

                    self._add_to_var(pop, var_name, times, series)
        self._row_count += 1

    def _collect_spikes(self, model):
        genn_model = model.network
        genn_model.pull_recording_buffers_from_device()
        for pop_name in self.recorded_vars:
            if "spikes" not in self.recorded_vars[pop_name]:
                continue
            genn_pop = model.connected_neurons[pop_name]
            spike_t = genn_pop.spike_recording_data[0]
            spike_id = genn_pop.spike_recording_data[1]
            self._add_to_var(pop_name, "spikes", spike_t, spike_id)

    def _add_to_var(self, pop, var, times, series):
        if var != "spikes":
            self._data[pop][var][self._row_count] = np.concatenate([times, series])
        else:
            self._data[pop][var] = np.column_stack([times, series])

    def _set_up_var_recording(self, to_be_tracked, connected_neurons):
        """
        Tracks a list of variables
        """
        res = {}
        with tables.open_file(self.logging_path, 'w', filters=self.filters) as f:
            for pop in to_be_tracked:
                group = f.create_group(f.root, pop)
                for var_name in to_be_tracked[pop]:
                    target_cols = self._get_cols_for_var(var_name, connected_neurons[pop])
                    expected_rows = self.simulation_time // (self.n_timesteps_to_pull_var * self.dt)
                    res.setdefault(pop, []).append(var_name)
                    f.create_earray(group, var_name, tables.Float64Atom(),
                                    (0, target_cols), expectedrows=expected_rows)
        return res

    def _get_cols_for_var(self, var_name: List[str], neurons):
        if var_name == "spikes":
            target_cols = 2
        else:
            # TODO: implement CUDA event buffers for those variables.
            # For now all we can do is pulling them manually every n steps

            # Format: (v_1, v_2, ..., v_k, t) with k being the number of neurons.
            target_cols = neurons.size + 1

            # For spikes, this is hopefully an upper bound. For vars, this is exact.
        return target_cols

    def _reset_population(self):
        self._row_count = 0

    def _stream_output(self):
        logging.info(f"Saving to {self.dirpath}")

        # This makes sure the table is locked until the simulation ends (graciously or not).
        if self._output_table is None:
            self._output_table = tables.open_file(self.logging_path, 'a', self.filters)

        for pop, var_dict in self._data.items():
            for var, values in var_dict.items():
                if len(values.shape) == 1:
                    values = values.reshape((1, -1))
                if var != "spikes":
                    values = values[:self._row_count]
                self._output_table.root[pop][var].append(values)

        self._reset_population()

    def flush(self):
        self._output_table.close()
        self._output_table = None
        self._reset_population()

    def record(self, model, save):
        """Saves the data to the disk

        Parameters
        ----------
        model : model.network.NeuronalNetwork
            The model for which data has to be saved
        save : bool
            Whether to save to disk
        """
        if not save:
            return
        if model.network.timestep % self.n_timesteps_to_pull_var == 0 or model.network.timestep == self.simulation_time / self.dt:
            self._collect_vars(model)

        if model.network.timestep % self.batch == 0 or model.network.timestep ==  self.simulation_time / self.dt:
            self._collect_spikes(model)
            self._stream_output()

    def dump_connectivity(self, model):
        """Writes the connectivity of the model to disk

        Parameters
        ----------
        model : model.network.NeuronalNetwork
            The model for which the connectivity
            has to be saved
        """

        connectivity = model.get_connectivity()
        connectivity = pd.DataFrame(connectivity)
        filename = self.dirpath / "connectivity.csv"
        connectivity.to_csv(filename)
