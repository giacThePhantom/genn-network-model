import numpy as np
import neuron
import synapse
from pygenn.genn_model import GeNNModel, GeNNType

class NeuronalNetwork:
    """
    Class containing the neuronal network definition

    ...

    Attributes
    ----------
    network : pygenn.genn_model.GeNNModel
        The genn object containing the network model
    neuron_populations : dict
        A dictionary containing all the different population of neurons
    synapses : dict
        A dictionary containing all the different synapses between the
        population of neurons

    Methods
    -------
    """

    network = None
    neuron_populations = {}
    connected_neurons = {}
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
            source, target = i.split("_")
            self.synapses[i] = synapse.Synapse(synapses[i],
                                               synapses[i]['name'],
                                               self.connected_neurons[synapses[i]['source']],
                                               self.connected_neurons[synapses[i]['target']],
                                               )

    def _connect(self):
        for i in self.synapses:
            self.connected_synapses[i] = self.synapses[i].add_to_network(self.network)


    def __init__(self, name, neuron_populations, synapses, dt, cuda_capable = True):
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
        """
        if cuda_capable:
            self.network = GeNNModel("double", name)
        else:
            self.network = GeNNModel("double", name, backend = "SingleThreadedCPU")

        self.network.dT = dt
        self._add_neuron_population(neuron_populations)
        self._add_neurons_to_network()

        self._add_synapses(synapses)
        self._connect()

    def build_and_load(self):
        """Builds the corresponding code (C++ or CUDA) and loads it for later use
        """
        self.network.build()
        self.network.load()

if __name__ == '__main__':
    import sys
    from reading_parameters import get_parameters
    params = get_parameters(sys.argv[1])
    model = NeuronalNetwork("Test", params['neuron_populations'], params['synapses'], 0.1)
    print("HERE")
    model.build_and_load()
    print(model.connected_neurons['or'].vars['ra_1'].view[:])
    model.network.step_time()
    print(model.connected_neurons['or'].vars['ra_1'].view[:])
    model.connected_neurons['or'].vars['ra_1'].view = np.arange(160)
    print(model.connected_neurons['or'].vars['ra_1'].view[:])
