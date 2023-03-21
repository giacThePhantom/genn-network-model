import numpy as np
from collections import defaultdict
import logging
from pathlib import Path
import pickle
import tables
from typing import List

class Recorder:
    """
    An object that records and saves data during a simulation run
    """

    def __init__(self, output_path, sim_name, to_be_tracked, connected_neurons, batch, batch_var_reads,  dt, simulation_time):

        self.dirpath = Path(output_path) / sim_name
        self.logging_path = self.dirpath / "tracked_vars.h5"
        self.protocol_path = self.dirpath / "protocol.pickle"
        self.dirpath.mkdir(exist_ok=True)
        self.filters = tables.Filters(complib='blosc:zstd', complevel=5)
        self._row_count = 0

        self.batch_var_reads = batch_var_reads
        self.batch_size_timesteps = round(batch / self.batch_var_reads)
        self.dt = dt
        self.simulation_time = simulation_time

        self.recorded_vars = self._set_up_var_recording(to_be_tracked, connected_neurons)

        self.recorded_vars = {}
        self._data = self._set_container_for_data_recording(self.recorded_vars, connected_neurons)
        self._output_table = None
        self.poll_spike_readings = True
        self.recorded_vars = {}
        pass

    def dump_protocol(self, protocol):
        with self.protocol_path.open('wb') as f:
            pickle.dump(protocol, f)

    def enable_spike_recording(self, model):
        for pop in self.recorded_vars:
            for var_name in self.recorded_vars[pop]:
                if var_name == 'spikes':
                    model.connected_neurons[pop].spike_recording_enabled = True

    def _set_container_for_data_recording(self, recorded_vars, connected_neurons):
        res = defaultdict(lambda : defaultdict(lambda : np.array([])))

        for pop in recorded_vars:
            for var_name in recorded_vars[pop]:
                if var_name != 'spikes':
                    res[pop][var_name] = np.empty((self.batch_size_timesteps, connected_neurons[pop].size + 1))

        return res


    def _collect_vars(self, model):
        for pop in self.recorded_vars:
            genn_pop = model.connected_neurons[pop]
            for var_name in self.recorded_vars[pop]:
                if var_name != 'spikes':
                    genn_pop.pull_var_name_from_device(var_name)
                    logging.debug(f"{pop} -> {var_name}")
                    logging.debug(genn_pop.var_names[var_name].view)
                    series = genn_pop.var_names[var_name].view.T
                    times = np.array([model.t])

                    self._add_to_var(pop, var_name, times, series)
        self._row_count += 1

    def _collect_spikes(self, poll_spike_readings, model):
        genn_model = model.network
        if not poll_spike_readings:
            genn_model.pull_recording_buffers_from_device()
        for pop_name in self.recorded_vars:
            if "spikes" not in self.recorded_vars[pop_name]:
                continue
            genn_pop = model.connected_neurons[pop_name]
            if not poll_spike_readings:
                spike_t = genn_pop.spike_recording_data[0]
                spike_id = genn_pop.spike_recording_data[1]
            else:
                genn_pop.pull_current_spikes_from_device()
                spike_count = genn_pop.spike_count[0][0]
                logging.debug(f"Detected {spike_count} spike events")
                if spike_count > 0:
                    spike_t = genn_model.t * np.ones(spike_count)
                    spike_id = genn_pop.spikes[0][0][:spike_count]
                else:
                    spike_t = []
                    spike_id = []
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
                    expected_rows = self.simulation_time // (self.batch_var_reads / self.dt)
                    f.create_earray(group, var_name, tables.Float64Atom(),
                                    (0, target_cols), expectedrows=expected_rows)
                    res.setdefault(pop, []).append(var_name)
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
                # handle both spiking events and snapshots
                if len(values.shape) == 1:
                    values = values.reshape((1, -1))
                self._output_table.root[pop][var].append(values)

        self._reset_population()

    def flush(self):
        self._output_table.close()
        self._output_table = None
        self._reset_population()

    def record(self, model, save):
        if model.network.timestep % self.batch_var_reads == 0:
            self._collect_vars(model)

        if model.network.timestep % self.batch_size_timesteps == 0:
            self._collect_spikes(self.poll_spike_readings, model)
            if save:
                self._stream_output()
