"""
This module implements multi-threading variable recording. Refer to `Recorder` for details
"""

from copy import deepcopy
from multiprocessing.pool import ThreadPool
import logging
from pathlib import Path
import pickle
import shutil
from typing import List

import numpy as np
import tables
import pandas as pd

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
    n_batches: int
        Number of batches in a simulation.
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

    def __init__(self, output_path, sim_name, to_be_tracked, connected_neurons, batch, n_timesteps_to_pull_var,  dt, simulation_time, n_processes=3):

        self.dirpath = Path(output_path) / sim_name
        # Horrible approach, but it should work:
        # generate a table for each dumped batch. Use a threadpool (say 3
        # threads, there's no point in having more) and make each thread write
        # onto such a pool. Once the simulation is over merge all slices in a
        # linear manner.
        self._logging_slices_builder = "tracked_vars_%.{}d.h5"
        self.logging_path = self.dirpath / "tracked_vars.h5"
        self.protocol_path = self.dirpath / "protocol.pickle"
        if self.dirpath.exists():
            shutil.rmtree(str(self.dirpath))
        self.dirpath.mkdir(exist_ok=True)
        self.filters = tables.Filters(complib='blosc:zstd', complevel=5)

        self.n_timesteps_to_pull_var = n_timesteps_to_pull_var
        self.n_points_in_batch = round(batch / self.n_timesteps_to_pull_var)
        self.batch = batch
        self.dt = dt
        self.simulation_time = simulation_time
        self.n_batches = int(np.floor(self.simulation_time // (self.batch * self.dt)))

        if self.n_batches > 1024:
            logging.warning("Up to %d partitions will be created this way! You should probably increase your batch size, or your (remote) file system performance will degrade." % self.n_batches)

        # Stateful variables
        # TODO these should be properly reset and/or restored, in case we wanted to implement checkpointing and/or resuming.
        self.recorded_vars = self._set_up_var_recording(to_be_tracked, connected_neurons)

        self._row_count = 0
        self._data = self._set_container_for_data_recording(self.recorded_vars, connected_neurons)
        self._cur_batch_no = 0
        self._pool = ThreadPool(processes=n_processes)
        self._futures = []

    def dump_protocol(self, protocol):
        events = deepcopy(protocol.events)
        for i in events:
            i.pop('binding_rates')
            i.pop('activation_rates')
            i.pop('happened')
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
                    logging.debug("%s -> %s", pop, var_name)
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

    def _get_slice_partition(self, slice_no) -> Path:
        # generates slice names based on how many batches we have
        # the batches are guaranteed to be lexicographically sorted through fixed digits
        digit_number = len(str(self.n_batches))
        builder_path = self._logging_slices_builder.format(digit_number)
        return self.dirpath / (builder_path % slice_no)

    def _set_up_var_recording(self, to_be_tracked, connected_neurons):
        """
        Tracks a list of variables
        """
        res = {}

        for batch in range(self.n_batches):
            partition_path = self._get_slice_partition(batch)

            with tables.open_file(partition_path, 'w', filters=self.filters) as f:
                for pop in to_be_tracked:
                    group = f.create_group(f.root, pop)
                    for var_name in to_be_tracked[pop]:
                        # FIXME maybe avoid recomputing these variables, they'll be the same in all batches.
                        target_cols = self._get_cols_for_var(var_name, connected_neurons[pop])
                        expected_rows = self.batch // self.n_timesteps_to_pull_var
                        recorded_vars = res.setdefault(pop, [])
                        if var_name not in recorded_vars:
                            recorded_vars.append(var_name)

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
        # Note: the GIL makes this whole code thread-safe: because we are in high-level land
        # there is no way two threads will execute this concurrently.
        partition_path = self._get_slice_partition(self._cur_batch_no)
        self._cur_batch_no += 1

        logging.info("Saving to %s ", partition_path)

        to_save = deepcopy(self._data)
        self._futures.append(self._pool.apply_async(self._unsafe_save, (partition_path, to_save, self._row_count, self.filters)))

        self._reset_population()

    @staticmethod
    def _unsafe_save(partition_path, data, row_count, filters):
        # Save the spike and normal polled data into the given partition
        logging.info("Inside unsafe: saving to %s", partition_path)
        with tables.open_file(partition_path, 'a', filters) as output_table:
            for pop, var_dict in data.items():
                for var, values in var_dict.items():
                    if len(values.shape) == 1:
                        values = values.reshape((1, -1))
                    if var != "spikes":
                        values = values[:row_count]
                    # This operation will release the GIL
                    output_table.root[pop][var].append(values)

        logging.info("Inside unsafe: %s closed", partition_path)


    def flush(self, model):
        self._reset_population()
        # There is a cleaner way to perform a join, probably
        for future in self._futures:
            future.get()

        self._merge_partitions(model.connected_neurons)

    def _merge_partitions(self, connected_neurons):
        paths = []
        with tables.open_file(self.logging_path, 'w', self.filters) as f:
            for pop in self.recorded_vars:
                group = f.create_group(f.root, pop)
                for var_name in self.recorded_vars[pop]:
                    target_cols = self._get_cols_for_var(var_name, connected_neurons[pop])
                    expected_rows = int(self.simulation_time / (self.dt * self.n_timesteps_to_pull_var))
                    earray = f.create_earray(group, var_name, tables.Float64Atom(),
                                    (0, target_cols), expectedrows=expected_rows)

                    for batch in range(self.n_batches):
                        paths.append(self._get_slice_partition(batch))
                        node = f'/{pop}/{var_name}'
                        logging.info("merging %s(%s) into %s", paths[-1], node, self.logging_path)
                        with tables.open_file(paths[-1], 'r') as old_partition:
                            earray.append(old_partition.root[node][:])
                        logging.info("merged")

        #for path in paths:
        #    path.unlink()

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

        if model.network.timestep % self.batch == 0 or model.network.timestep == self.simulation_time / self.dt:
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
