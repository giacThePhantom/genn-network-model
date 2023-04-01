from . import neuron
from . import synapse
from . import draw_connectivity

from typing import Dict
import numpy as np
from pygenn.genn_model import GeNNModel, NeuronGroup

class NeuronalNetwork:
    """
    Class containing the neuronal network definition

    ...

    Attributes
    ----------
    network : pygenn.genn_model.GeNNModel
        The gen object containing the network model
    neuron_populations : dict
        A dictionary containing all the different population of neurons
    synapses : dict
        A dictionary containing all the different synapses between the
        population of neurons

    Methods
    -------
    """

    # network = None
    neuron_populations = {}
    connected_neurons: Dict[str, NeuronGroup] = {}
    synapses = {}
    connected_synapses = {}

    def _add_neuron_population(self, neuron_populations):
        """Reads the neuron_population dictionary and creates corresponding
           NeuronPopulation objects
        Parameters
        ----------
        neuron_population : dict
            A dictionary containing all the parameters necessary to build
            the population of neurons.
        """
        for i in neuron_populations:
            self.neuron_populations[i] = neuron.NeuronPopulation(neuron_populations[i], i)

    def _add_neurons_to_network(self):
        """Adds the NeuronPopulation to the GeNNModel"""
        for i in self.neuron_populations:
            if self.neuron_populations[i].size() > 0:
                self.connected_neurons[i] = self.neuron_populations[i].add_to_network(self.network)


    def _add_synapses(self, synapses):
        for i in synapses:
            self.synapses[i] = synapse.Synapse(synapses[i],
                                               synapses[i]['name'],
                                               self.connected_neurons[synapses[i]['source']],
                                               self.connected_neurons[synapses[i]['target']],
                                               )

    def _connect(self):
        for i in self.synapses:
            self.connected_synapses[i] = self.synapses[i].add_to_network(self.network)


    def __init__(self, name, neuron_populations, synapses, dt, cuda_capable = True, **backend_kwargs):
        """Builds a NeuronalNetwork object starting from a dictionary of
           neurons and one of synapses
        Parameters
        ----------
        name : str
            The unique identifier for the network model
        neuron_populations : dict
            A dictionary containing all the neuron populations
        synapses : dict
            A dictionary containing all the synapses
        cuda_capable : bool
            Whether the model has to be uploaded to a cuda device or it has to
            remain into a single threaded cpu process
        backend_kwargs: pass backend options. Refer to PreferenceBase
        ( https://genn-team.github.io/genn/documentation/4/html/d1/d7a/structCodeGenerator_1_1PreferencesBase.html ) for details
        """
        print(backend_kwargs)
        if cuda_capable:
            self.network = GeNNModel("double", name, **backend_kwargs)
        else:
            self.network = GeNNModel("double", name, backend = "SingleThreadedCPU", **backend_kwargs)

        self.network.dT = dt
        self._add_neuron_population(neuron_populations)
        self._add_neurons_to_network()

        self._add_synapses(synapses)
        self._connect()

    def build_and_load(self, num_recording_steps=None):
        """Builds the corresponding code (C++ or CUDA) and loads it for later use

        Parameters
        ----------
        num_recording_steps: Optional[int]
            When recording, this provides the size of the internal buffer
            used to keep track of the events (if using event batching). The user *must* pull from the
            event buffer once it is full, not any earlier (and possibly not further or it will fill up)
        """
        self.network.build()
        self.network.load(num_recording_timesteps=num_recording_steps)

    def reinitialize(self):
        """
        Reset the internal model variables and clear all currents logs
        """
        self.network.reinitialise()

    def preallocate_logs(self, pop, var):
        # Wrapper around NeuronPopulation.preallocate_logs
        self.neuron_populations[pop].preallocate_logs(var)

    def get_connectivity(self):
        res = []
        for i in self.connected_synapses:
            if self.connected_synapses[i].is_ragged:
                self.connected_synapses[i].pull_connectivity_from_device()
                for (j, z) in zip(self.connected_synapses[i].get_sparse_pre_inds(), self.connected_synapses[i].get_sparse_post_inds()):
                    res.append({
                        "pre_population" : self.connected_synapses[i].src.name,
                        "post_population" : self.connected_synapses[i].trg.name,
                        "pre_id" : j,
                        "post_id" : z,
                    })
            else:
                source_size = self.connected_synapses[i].src.size
                target_size = self.connected_synapses[i].trg.size
                connections = self.connected_synapses[i].get_var_values('g').reshape(source_size, target_size)
                for (j, row) in enumerate(connections):
                    for z in range(len(row)):
                        if z != 0:
                            res.append({
                                "pre_population" : self.connected_synapses[i].src.name,
                                "post_population" : self.connected_synapses[i].trg.name,
                                "pre_id" : j,
                                "post_id" : z,
                            })

        return res


if __name__ == '__main__':
    import sys
    from reading_parameters import get_parameters
    params = get_parameters(sys.argv[1])
    model = NeuronalNetwork("Test", params['neuron_populations'], params['synapses'], 0.1)
    model.build_and_load()
    model.network.step_time()
    draw_connectivity.create_glomerulus_graph(model, int(sys.argv[2]), sys.argv[3])
