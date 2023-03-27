import numpy as np
import logging
from pathlib import Path
import pickle
import tables
from typing import List
import threading
import queue

class Recorder:
    """
    An object that records and saves data during a simulation run
    """

    def disk_writer(self, queue, table_path, to_be_tracked, connected_neurons):
        filters = tables.Filters(complib='blosc:zstd', complevel=5)
        output_table = tables.open_file(table_path, 'w', filters)
        print(table_path)
        print(to_be_tracked)
        for pop in to_be_tracked:
            group = output_table.create_group(output_table.root, pop)
            for var_name in to_be_tracked[pop]:
                target_cols = self._get_cols_for_var(var_name, connected_neurons[pop])
                expected_rows = self.simulation_time // (self.n_timesteps_to_pull_var * self.dt)
                output_table.create_earray(group, var_name, tables.Float64Atom(),
                                (0, target_cols), expectedrows=expected_rows/2)
        while True:
            to_be_written = queue.get()

            for [pop, var, times, series] in to_be_written:

                if var == 'spikes':
                    values = np.column_stack([times, series])
                else:
                    values = np.concatenate([times, series])


                if len(values.shape) == 1:
                    values = values.reshape((1, -1))


                output_table.root[pop][var].append(values)
            queue.task_done()

    def __init__(self, output_path, sim_name, to_be_tracked, connected_neurons, batch, n_timesteps_to_pull_var,  dt, simulation_time):

        self.dirpath = Path(output_path) / sim_name
        self.logging_path = [self.dirpath / f"tracked_vars_{i}.h5" for i in range(2)]
        self.protocol_path = self.dirpath / "protocol.pickle"
        self.dirpath.mkdir(exist_ok=True)

        self.n_timesteps_to_pull_var = n_timesteps_to_pull_var
        self.n_points_in_batch = round(batch / self.n_timesteps_to_pull_var)
        self.dt = dt
        self.simulation_time = simulation_time

        self.recorded_vars = self._set_up_var_recording(to_be_tracked, connected_neurons)

        self._row_count = 0
        self._data = self._set_container_for_data_recording(self.recorded_vars, connected_neurons)
        self._output_table = []
        self.queue = queue.Queue(maxsize = batch)
        for i in self.logging_path:
            threading.Thread(target=self.disk_writer, args = (self.queue, str(i), to_be_tracked, connected_neurons), daemon=True).start()


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
        res = {}
        for pop in recorded_vars:
            res[pop] = {}
            for var_name in recorded_vars[pop]:
                if var_name != 'spikes':
                    res[pop][var_name] = np.empty((self.n_points_in_batch, connected_neurons[pop].size + 1))

        return res



    def _set_up_var_recording(self, to_be_tracked, connected_neurons):
        """
        Tracks a list of variables
        """
        res = {}
        for pop in to_be_tracked:
            for var_name in to_be_tracked[pop]:
                res.setdefault(pop, []).append(var_name)

        # for i in self.logging_path:
        #     with tables.open_file(str(i), 'w', filters=self.filters) as f:
        #         for pop in to_be_tracked:
        #             group = f.create_group(f.root, pop)
        #             for var_name in to_be_tracked[pop]:
        #                 target_cols = self._get_cols_for_var(var_name, connected_neurons[pop])
        #                 expected_rows = self.simulation_time // (self.n_timesteps_to_pull_var * self.dt)
        #                 f.create_earray(group, var_name, tables.Float64Atom(),
        #                                 (0, target_cols), expectedrows=expected_rows)
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

        for pop, var_dict in self._data.items():
            for var, values in var_dict.items():
                # handle both spiking events and snapshots
                if len(values.shape) == 1:
                    values = values.reshape((1, -1))
                self._output_table.root[pop][var].append(values[:self._row_count])

        self._reset_population()

    def flush(self):
        print("Joining")
        self.queue.join()

    def record(self, model, save):
        to_be_written = []
        if model.network.timestep % self.n_timesteps_to_pull_var == 0 or model.network.timestep == self.simulation_time / self.dt:
            to_be_written = self._collect_vars(model)

        if model.network.timestep % self.n_points_in_batch == 0 or model.network.timestep ==  self.simulation_time / self.dt:
            to_be_written += self._collect_spikes(model)

        if save and len(to_be_written) > 0:
            self.queue.put(to_be_written)


    def _collect_vars(self, model):
        res = []
        for pop in self.recorded_vars:
            genn_pop = model.connected_neurons[pop]
            for var_name in self.recorded_vars[pop]:
                if var_name != 'spikes':
                    genn_pop.pull_var_from_device(var_name)
                    logging.debug(f"{pop} -> {var_name}")
                    logging.debug(genn_pop.vars[var_name].view)
                    series = genn_pop.vars[var_name].view.T
                    times = np.array([model.network.t])

                    res.append([pop, var_name, times, series])
        self._row_count += 1
        return res

    def _collect_spikes(self, model):
        genn_model = model.network
        genn_model.pull_recording_buffers_from_device()
        res = []
        for pop_name in self.recorded_vars:
            if "spikes" not in self.recorded_vars[pop_name]:
                continue
            genn_pop = model.connected_neurons[pop_name]
            spike_t = genn_pop.spike_recording_data[0]
            spike_id = genn_pop.spike_recording_data[1]
            res.append([pop_name, "spikes", spike_t, spike_id])
        return res

    def _add_to_var(self, pop, var, times, series):
        # if var != "spikes":
        #     self._data[pop][var][self._row_count] = np.concatenate([times, series])
        # else:
        #     self._data[pop][var] = np.column_stack([times, series])
        #
        self.queue.put([pop, var, times, series])
        pass
