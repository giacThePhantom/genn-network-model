import tables
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

class DataManager:
    """A class which deals with reading the data that a simulation
    saves to disk, reading it and performing operation on it so to
    provide it in a form for easy plotting

    Attributes
    ----------
    sim_param : dict
        The parameters that determine the simulation
    sim_name : str
        The name of the simulation
    neuron_param : dict
        The parameters that characterize the neuron populations in
        the model
    synapse_param : dict
        The parameters that characterize the synapse populations in
        the model
    data : tables.File
        The output of a simulation
    recorded_data : dict
        Which variables are present in the file
    events : pandas.DataFrame
        A dataframe containing all the events that happened during
        the simulation
    protocol : Protoocl
        The protocol used to generate events during the simulation

    Methods
    -------

    close() : None
        Closes the data file, to be used when the
        object is no longer referenced
    get_data_window(var_path, t_start, t_end) : numpy.ndArray
        Get all the collected data for a given variable from time
        t_start to time t_end
    get_data_for_first_neuron_in_glomerulus(glo_idx, pop, var, t_start, t_end) : numpy.ndArray
        Gets the evolution of a variable during a time interval for the first
        neuron of a population in a glomerulus
    get_spikes_for_first_neuron_in_glomerulus(glo_idx, pop, t_start, t_end) : numpy.ndArray
        Gets the time at which spikes happen for the first neuron of a population
        in a glomerulus during a time step
    get_first_neuron_in_glomerulus(glo_idx, pop) : int
        Returns the first neuron of a population in a glomerulus
    or_most_active(ra) : int
        Gets the most active glomerulus given the ra matrix
    get_spike_matrix(spike_times, spike_ids, pop, t_start, t_end) : numpy.ndArray
        Compute the matrix of spikes for a population in an interval
    sdf_for_population(pop, t_start, t_end) : numpy.ndArray
        Computes the spike density matrix for a population
        in a time interval
    sdf_per_glomerulus_avg(pop, t_start, t_end) : numpy.ndArray
        The average sdf in each glomerulus for each
        time step during a time interval.
    sdf_time_avg(sdf) : numpy.ndArray
        Takes a sdf matrix and computes the corresponding
        time average
    get_active_glomeruli_per_pop(sdf) : numpy.ndArray
        Given a spike density matrix computes which glomeruli
        are activated (or deactivated) as if their distance from
        the mean value of sdf is more than one standard deviation
    get_sim_dt() : double
        Getter for the timestep of the simulation
    show_or_save(filename, show)
        Whether to show or save a matplotlibe image
    get_pop_size(pop) : int
        Getter for the population size of a neuron
        population
    get_inhibitory_connectivity() : numpy.ndArray
        Getter for the inhibitory connectivity
    get_events() : pandas.DataFrame
        Getter for the events that happened during the
        simulation
    get_sim_name() : str
        Getter for the simulation name
    """

    def __init__(self, sim_param, sim_name, neuron_param, synapse_param):
        """Builds the data manager object

        Parameters
        ----------
        sim_param : dict
            The parameter of the simulation
        sim_name : str
            The name of the simulation
        neuron_param : dict
            The parameters that characterize the neuron populations in
            the model
        synapse_param : dict
            The parameters that characterize the synapse populations in
            the model
        """

        self.sim_param = sim_param
        self.neuron_param = neuron_param
        self.synapse_param = synapse_param
        self.sim_name = sim_name
        self._root_out_dir = Path(sim_param['output_path']) / sim_name
        self._root_plot_dir = self._root_out_dir / 'plots'
        self._root_plot_dir.mkdir(exist_ok=True)
        self.data = tables.open_file(
            str(self._root_out_dir / 'tracked_vars.h5'))
        self.recorded_data = sim_param['tracked_variables']
        self.events = pd.read_csv(self._root_out_dir / 'events.csv')
        with (self._root_out_dir / 'protocol.pickle').open('rb') as f:
            self.protocol = pickle.load(f)

    def close(self):
        """Closes the data file, to be used when
        destructing this object
        """
        self.data.close()

    def get_data_window(self, var_path, t_start, t_end):
        """Get all the collected data for a given variable from time
        t_start to time t_end

        Parameters
        ----------
        var_path : list
            Where the variable has been saved in the output file
            (usually neuron_pop/name)
        t_start : float
            The start of the interval of interest
        t_end : float
            The end of the interval of interest

        Returns
        -------
        filtered_timesteps : numpy.ndArray
            An array containing all the timesteps in the time interval
        filtered_data : numpy.ndArray
            An array containing the value of the variabile in the time
            interval
        """

        path = '/' + '/'.join(var_path)
        data = self.data.root[path]
        if var_path[-1] == 'spikes':
            # horrible hack to read the entire EArray as a numpy array.
            copied_data = np.squeeze(data[:])
            timesteps = data[:, 0]
            timesteps_start = timesteps >= t_start
            timesteps_end = timesteps < t_end
            filtered = np.where(timesteps_start & timesteps_end)
            filtered_timesteps = np.squeeze(copied_data[filtered, 0])
            filtered_data = np.squeeze(copied_data[filtered, 1:])

            return filtered_timesteps, filtered_data

        else:
            timestep_start = np.floor(
                t_start/(self.sim_param['dt']*self.sim_param['n_timesteps_to_pull_var']))
            timestep_end = np.ceil(
                t_end/(self.sim_param['dt']*self.sim_param['n_timesteps_to_pull_var']))
            data = data[timestep_start:timestep_end]
            return data[:, 0], np.squeeze(data[:, 1:])

    def get_data_for_first_neuron_in_glomerulus(self, glo_idx, pop, var, t_start, t_end):
        """Gets the evolution of a variable during a time interval for the first
        neuron of a population in a glomerulus

        Parameters
        ----------
        glo_idx : int
            The index of the glomerulus of interest
        pop : str
            The population of neurons of interest
        var : str
            The variable of interest
        t_start : float
            The start of the time interval
        t_end : float
            The end of the time interval

        Returns
        -------
        time : numpy.ndArray
            An array containing all the timestep in which the
            variable has been recorded
        var_value : numpy.ndArray
            An array containing the variable in the interval
            for the first neuron in the glomerulus
        """

        neuron_idx = self.get_first_neuron_in_glomerulus(glo_idx, pop)
        time, var_value = self.get_data_window((pop, var), t_start, t_end)
        var_value = var_value[:, neuron_idx]
        return time, var_value

    def get_spikes_for_first_neuron_in_glomerulus(self, glo_idx, pop, t_start, t_end):
        """Gets the time at which spikes happen for the first neuron of a population
        in a glomerulus during a time step

        Parameters
        ----------
        glo_idx : int
            The index of the glomerulus of interest
        pop : str
            The population of neurons of interest
        t_start : float
            The start of the time interval
        t_end : float
            The end of the time interval

        Returns
        -------
        spike_times : numpy.ndArray
            An array containing all the timestep in which the
            neuron has spiked
        """

        neuron_idx = self.get_first_neuron_in_glomerulus(glo_idx, pop)
        spike_times, spike_id = self.get_data_window(
            (pop, "spikes"), t_start, t_end)
        filtered_spike_idx = spike_id == neuron_idx
        spike_times = spike_times[filtered_spike_idx]
        return spike_times

    def get_first_neuron_in_glomerulus(self, glo_idx, pop):
        """Returns the first neuron of a population in a glomerulus

        Parameters
        ----------
        glo_idx : int
            The index of the glomerulus of interest
        pop : str
            The population of neurons of interest

        Returns
        -------
        neuron_idx : int
            The index for the first glomerulus in a
            population
        """

        neuron_idx = glo_idx * \
            self.neuron_param[pop]['n'] // self.neuron_param['or']['n']
        return neuron_idx

    def or_most_active(self, ra):
        """Gets the most active glomerulus given the ra matrix

        Parameters
        ----------
        ra : numpy.ndArray
            The matrix containing the ra variable
        Returns
        -------
        or_most_active : int
            The index of the most active glomerulus
        """

        #Sum over time
        ra_sum = np.sum(ra, axis=0)
        #Pick the or that has been the most active during
        #The time window
        or_most_active = np.argmax(ra_sum)
        return or_most_active

    def get_spike_matrix(self, spike_times, spike_ids, pop, t_start, t_end):
        """Compute the matrix of spikes for a population in an interval

        Parameters
        ----------
        spike_times : numpy.ndArray
            The times at which a spike has appened
        spike_ids : numpy.ndArray
            The ids of which neuron has spiked during the interval
        pop : str
            The population to consider
        t_start : float
            The start of the time interval
        t_end : float
            The end of the time interval

        Returns
        -------
        res : numpy.ndArray
            A matrix with as a first dimension the dimension of the
            population and as the second the number of timesteps.
            res[i,j] = 1 if neuron i has spiked at time j, 0 otherwise
        """

        duration_timesteps = int(
            np.ceil((t_end-t_start)/self.sim_param['dt'])) + 1
        res = np.zeros((self.neuron_param[pop]['n'], duration_timesteps))

        for (time, id) in zip(spike_times, spike_ids):
            time = int((time - t_start)/self.sim_param['dt'])
            res[int(id)][time] = 1.0

        return res

    def sdf_for_population(self, pop, t_start, t_end):
        """Computes the spike density matrix for a population
        in a time interval

        Parameters
        ----------
        pop : str
            The population of interest
        t_start : float
            The start of the time interval
        t_end : float
            The end of the time interval

        Returns
        -------
        sdf : numpy.ndArray
            The spike density matrix
        """

        spike_times, spike_ids = self.get_data_window(
                (pop, 'spikes'),
                t_start,
                t_end,
                )
        spike_matrix = self.get_spike_matrix(
                spike_times,
                spike_ids,
                pop,
                t_start,
                t_end
                )
        sigma = self.sim_param['sdf_sigma']
        dt = self.sim_param['dt']
        kernel = np.arange(-3*sigma, +3*sigma, dt)
        kernel = np.exp(-np.power(kernel, 2)/(2*sigma*sigma))
        kernel = kernel/(sigma*np.sqrt(2.0*np.pi))*1000.0
        sdf = np.apply_along_axis(
                lambda m : sp.signal.convolve(m, kernel, mode='same'),
                axis = 1,
                arr=spike_matrix)
        return sdf

    def sdf_per_glomerulus_avg(self, pop, t_start, t_end):
        """The average sdf in each glomerulus for each
        time step during a time interval.

        Parameters
        ----------
        pop : str
            The population of interest
        t_start : float
            The start of the time interval
        t_end : float
            The end of the time interval

        Returns
        -------
        res : numpy.ndArray
            The spike density matrix averaged over
            neurons in a glomerulus
        """
        sdf = self.sdf_for_population(pop, t_start, t_end)
        n_glomeruli = self.neuron_param['or']['n']
        res = np.zeros((n_glomeruli, sdf.shape[1]))
        glomerulus_dim = sdf.shape[0] // n_glomeruli
        for i in range(n_glomeruli):
            res[i, :]= np.mean(sdf[glomerulus_dim*i:glomerulus_dim*(i+1), :],axis=0)
        return res

    def sdf_time_avg(self, sdf):
        """Takes a sdf matrix and computes the corresponding
        time average

        Parameters
        ----------
        sdf : numpy.ndArray
            The spike density matrix

        Returns
        -------
        res : numpy.ndArray
            The spike density matrix averaged over time
        """

        return sdf.mean(axis = 1)

    def get_active_glomeruli_per_pop(self, sdf):
        """Given a spike density matrix computes which glomeruli
        are activated (or deactivated) as if their distance from
        the mean value of sdf is more than one standard deviation

        Parameters
        ----------
        sdf : numpy.ndArray
            The spike density matrix

        Returns
        -------
        glomeruli_of_interest : list
            The list of indices of activated or deactivated
            neurons
        """

        glomeruli_mean_sdf = np.mean(sdf, axis = 1)
        global_mean = np.mean(glomeruli_mean_sdf)
        global_sdt = np.std(glomeruli_mean_sdf)
        glomeruli_of_interest = []

        for (i, mean_sdf) in enumerate(glomeruli_mean_sdf):
            if np.abs(mean_sdf - global_mean) > global_sdt:
                glomeruli_of_interest.append(i)
        return glomeruli_of_interest

    def sdf_correlation(self, sdf):
        """Computes the correlation matrix of the sdf

        Parameters
        ----------
        sdf : numpy.ndArray
            The spike density matrix

        Returns
        -------
        res : numpy.ndArray
            The correlation matrix
        """

        return np.corrcoef(sdf)

    def get_sim_dt(self):
        """Getter for the timestep of the simulation

        Returns
        -------
        res : float
            The timestep of the simulation
        """

        return self.sim_param['dt']

    def show_or_save(self, filename, show=False):
        """Whether to show or save a matplotlibe image

        Parameters
        ----------
        filename : pathlib.Path
            Where to save the image
        show : boolt
            Whether to save or show the image
        """

        filename = self._root_plot_dir / filename
        filename.parent.mkdir(exist_ok=True)
        if show:
            plt.show()
        else:
            print("Saving to ", filename)
            plt.savefig(filename, dpi=700, bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

    def get_pop_size(self, pop):
        """Getter for the population size of a neuron
        population

        Parameters
        ----------
        pop ; str
            The name of the population

        Returns
        -------
        res : int
            The number of neurons in the population
        """

        return self.neuron_param[pop]['n']

    def get_inhibitory_connectivity(self):
        """Getter for the inhibitory connectivity

        Returns
        -------
        connectivity_matrix : numpy.ndArray
            The matrix containing the inter-glomeruli connectivity,
            with shape (n_or, n_or)
        """

        connectivity_matrix = self.protocol._generate_inhibitory_connectivity(self.protocol.param['connectivity_type'], self.protocol.param['self_inhibition'])
        return connectivity_matrix

    def get_events(self):
        """Getter for the events that happened during the
        simulation

        Returns
        -------
        res : pandas.DataFrame
            The dataframe containing all the events
        """

        return self.events

    def get_sim_name(self):
        """Getter for the simulation name

        Returns
        -------
        res : str
            The name of the simulation
        """

        return self.sim_name
