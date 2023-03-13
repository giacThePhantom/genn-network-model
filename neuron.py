import numpy as np
from pygenn.genn_model import create_custom_neuron_class

class NeuronPopulation:
    """
    Build a custom neuron population type and define its parameters, so to
    Interface it with a network model later on

    ...

    Attributes
    ----------
    name : str
        Unique name for the neuron population
    n : int
        Number of neurons in the population
    neuron_class : NeuronModels::Custom
        The custom neuron class built by genn
    parameters : dict
        Dictionary containing {parameter_name : value} for all the parameters necessary
    variables : list
        List containing tuples (variable_name, precision(scalar for global)) for
        the variables describing the dynamic behaviour of the neuron.
    initial_variables : dict
        Dictionary containing {variable_name : initial_value} for the initial values
        of neurons' variables.
    spike_rec : bool
        Whether the simulation has to record spikes for this type of neurons
    eqs : str
        Contains the equation for the time evolution of the neuron
    reset_code : str
        How to reset when a spike happens
    threshold_condition_code : str
        Condition to when to execute reset_code

    Methods
    -------
    create_population()
        Returns all the values necessary to Genn to crate a population of neurons
    add_to_network(network)
        Takes as input a genn network model and adds this population of neurons to it
    size()
        Returns the number of neurons in the population
    """

    name = ""
    n = 0
    neuron_class = None
    parameters = {}
    variables = []
    initial_variables = {}
    spike_rec = False
    eqs = ""
    reset_code = ""
    threshold_condition_code = ""
    recorded_outputs = {}

    def _reset(self):
        self.name = ""
        self.n = 0
        self.neuron_class = None
        self.parameters = {}
        self.variables = []
        self.initial_variables = {}
        self.spike_rec = False
        self.eqs = ""
        self.reset_code = ""
        self.threshold_condition_code = ""

        # each of these should be a 2xE matrix with E being the total number of events.
        # Because the semantics of event changes depending on the variable,
        # not all tensors may coincide in size.
        self.recorded_outputs = {}

    def __init__(self, params, name):
        """Builds a neuron object from a dictionary containing all its parameters
        Parameters
        ----------
        params : dict
            The dictionary containing all the parameters necessary to build the neuron
        name : str
            The unique identifier of the neuron population
        """
        self._reset()
        self.name = name
        self.n = params['n']
        self.spike_rec = params['spike_rec']
        self.parameters = params['parameters']

        for var in params['variables']:
            self.variables.append((var['name'],  var['type']))
            if not isinstance(var['value'], list):
                var['value'] = [var['value'] for i in range(self.n)]
            self.initial_variables[var['name']] = var['value']

        self.eqs = "\n".join(params['sim_code'])
        self.reset_code = "\n".join(params['reset_code'])
        self.threshold_condition_code = "\n".join(params['threshold_condition_code'])

        self.neuron_class = create_custom_neuron_class(self.name + '_model',
                                                     param_names = list(self.parameters.keys()),
                                                     var_name_types = self.variables,
                                                     sim_code = self.eqs,
                                                     reset_code = self.reset_code,
                                                     threshold_condition_code = self.threshold_condition_code
                                                     )

    def __str__(self):
        """Convert the object to a string to facilitate debug
        Return
        ------
        res : string
            The Neuron object converted into string.
        """
        res = "Neuron:\n\n"
        res += "name:\n\t" +  str(self.name) + '\n\n'
        res += "population:\n\t" + str(self.n) + '\n\n'
        res += "Recording spikes::\n\t" + str(self.spike_rec) + '\n\n'
        res += "Parameters:\n"
        temp_param = list(self.parameters.items())
        for i in range(0, len(temp_param), 2):
            res += '\t' + str(temp_param[i][0]) + ': ' + str(temp_param[i][1])
            if i+1 < len(temp_param):
                res += '\t\t' + str(temp_param[i+1][0]) + ': ' + str(temp_param[i+1][1]) + '\n'
        res += "\n"
        res += "Variables:\n"
        for i in range(0, len(self.variables), 2):
            res += '\t' + str(self.variables[i][0]) + ': ' + str(self.variables[i][1])
            if i+1 < len(self.variables):
                res += '\t\t' + str(self.variables[i+1][0]) + ': ' + str(self.variables[i+1][1]) + '\n'
        res += "\n"
        res += "Initial values for variables:\n"
        temp_initial_variables = list(self.initial_variables.items())
        for i in range(0, len(temp_initial_variables), 2):
            res += '\t' + str(temp_initial_variables[i][0]) + ': ' + str(temp_initial_variables[i][1])
            if i+1 < len(temp_initial_variables):
                res += '\t\t' + str(temp_initial_variables[i+1][0]) + ': ' + str(temp_initial_variables[i+1][1]) + '\n'
        res += "\n"
        res += "Equation for simulation:\n\t" + str(self.eqs).replace("\n", "\n\t") + '\n\n'
        res += "What to do when threshold is crossed:\n\t" + str(self.reset_code).replace("\n", "\n\t") + '\n\n'
        res += "When a threshold is crossed:\n\t" + str(self.threshold_condition_code) + '\n\n'
        res += "Custom neuron:\n\t" + str(self.neuron_class) + '\n\n'
        return res

    def create_population(self):
        """Returns all the values necessary to genn to create a neuron population as a list
        Return
        ------
        res : list
            A list containing all the values to create a neuron population for genn
        """
        return [self.name,
                 self.n,
                 self.neuron_class,
                 self.parameters,
                 self.initial_variables
                 ]

    def add_to_network(self, network):
        """Add the neuron population to a network
        Parameters
        ---------
        network : pygenn.genn_model.GeNNModel
            The network model to which the neuron population is added, with
            whether the population has to be recorded
        res : list
            A list containing all the values to create a neuron population for genn
        """
        res = network.add_neuron_population(self.name,
                                            self.n,
                                            self.neuron_class,
                                            self.parameters,
                                            self.initial_variables
                                            )
        res.spike_recording_enabled = self.spike_rec
        return res

    def size(self):
        """Returns the number of neurons in the population
        Return
        ------
        res : int
            The number of neurons in the population
        """
        return self.n


if __name__ == '__main__':
    import sys
    from reading_parameters import get_parameters
    params = get_parameters(sys.argv[1])
    orn = NeuronPopulation(params['neuron_populations']['orn'], 'orn')
    #print(orn)
