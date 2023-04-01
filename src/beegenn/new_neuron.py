from pygenn.genn_model import create_custom_neuron_class, create_dpf_class

precision_types = ['float', 'double', 'scalar']


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

    def _validate_variables(self, param):
        param['var_space'] = {i['name']:i['value'] for i in param['model_variables']}
        if isinstance(param['neuron_class'], dict):
            param['neuron_class']['var_name_types'] = [(i['name'], i['type']) for i in param['model_variables']]

    def _validate_custom_neuron_class(self, param):
        if not isinstance(param['class_name'], str) or len(param['class_name']) == 0:
            raise Exception(f'Object NeuronPopulation requires a not null unique id for the custom neuron class in neuron_class_class_name, you provided {params["class_name"]}')
        if not isinstance(param['param_names'], list):
            raise Exception(f'Object NeuronPopulation requires a list of names of parameters for the custom neuron_class, you provided {params["param_names"]}')
        else:
            for i in param['param_names']:
                if not isinstance(i, str):
                    raise Exception(f'Object NeuronPopulation requires parameter names as string for the custom neuron_class, you provided {i}')
        if not isinstance(param['var_name_types'], list):
            raise Exception(f'Object NeuronPopulation requires a list of ("names", "precision") of variables for the custom neuron_class, you provided {params["var_name_types"]}')
        else:
            for i in param['var_name_types']:
                if not isinstance(i[0], str):
                    raise Exception(f'Object NeuronPopulation requires variable names in ("names", "precision") as string for the custom neuron_class, you provided {i[0]}')
                if not i[1] in precision_types:
                    raise Exception(f'Object NeuronPopulation requires variable precision in ("names", "precision") as one of {precision_types} for the custom neuron_class, you provided {i[1]}')
        if not isinstance(param['derived_params'], list):
            raise Exception(f'Object NeuronPopulation requires derived parameters as a dictionary of "{" name : lambda function"}" as the derived_params for the custom neuron_class, you provided {param["derived_params"]}')
        else:
            for i in param['derived_params']:
                if not isinstance(i[0], str):
                    raise Exception(f'Object NeuronPopulation requires derived parameters name as a string as the name for a derived parameter for the custom neuron_class, you provided {i}')
                if not callable(i[1]):
                    raise Exception(f'Object NeuronPopulation requires derived parameters value as a function as the value for a derived parameter for the custom neuron_class, you provided {param["derived_params"][i]}')

        self._validate_complex_string_parameters(param, 'sim_code')
        self._validate_complex_string_parameters(param, 'threshold_condition_code')
        self._validate_complex_string_parameters(param, 'reset_code')
        self._validate_complex_string_parameters(param, 'support_code')

        if not isinstance(param['extra_global_params'], list):
            raise Exception(f'Object NeuronPopulation requires extra global parameters as a list of ("name", "type"), for the custom neuron_class, you provided {params["extra_global_params"]}')
        else:
            for i in param['extra_global_params']:
                if not isinstance(i[0], str) or len(i[0]) == 0:
                    raise Exception(f'Object NeuronPopulation requires extra global parameters names to be a non empty string for the custom neuron_class, you provided {i[0]}')
                if not i[1] in precision_types:
                    raise Exception(f'Object NeuronPopulation requires extra global parameters type to be one of {precision_types} for the custom neuron_class, you provided {i[1]}')

        if not isinstance(param['additional_input_vars'], list):
            raise Exception(f'Object NeuronPopulation requires additional input variables as a list of ("name", "type", initial_value), for the custom neuron_class, you provided {params["additional_input_vars"]}')
        else:
            for i in param['additional_input_vars']:
                if not isinstance(i[0], str) or len(i[0]) == 0:
                    raise Exception(f'Object NeuronPopulation requires additional input variables names to be a non empty string for the custom neuron_class, you provided {i[0]}')
                if not i[1] in precision_types:
                    raise Exception(f'Object NeuronPopulation requires additional_input_vars type to be one of {precision_types} for the custom neuron_class, you provided {i[1]}')
                try:
                    tmp = int(i[2])
                except:
                    raise Exception(f'Object NeuronPopulation requires additional_input_vars initial value to be a number for the custom neuron_class, you provided {i[2]}')

        if not isinstance(param['is_auto_refractory_required'], bool):
            raise Exception(f'Object NeuronPopulation requires is_auto_refractory_required to be a boolean value for the custom neuron_class, you provided {i[2]}')

            #Dictionary with additional attributes and methods of the new class
        if not isinstance(param['custom_body'], dict):
            raise Exception(f'Object NeuronPopulation requires custom_body to be a dictionary containing as keys the name of the new method and as value a string representing the code for it for the custom neuron_class, you provided {param["custom_body"]}')

    def _validate_complex_string_parameters(self, param, name):
        if isinstance(param[name], list):
            param[name] = "\n".join(param[name])
        if not isinstance(param[name], str):
            raise Exception(f'Object NeuronPopulation requires {name} to be a string or a list of strings for the custom neuron, you provided {param[name]}')

    def _validate_input(self, params):
        self._validate_variables(params)
        if not isinstance(params['pop_name'], str) or len(params['pop_name']) == 0:
            raise Exception(f'Object NeuronPopulation requires a not null unique id for the population in pop_name, you provided {params["pop_name"]}')
        if not isinstance(params['num_neurons'], int) or params['num_neurons'] < 0:
            raise Exception(f'Object NeuronPopulation requires an integer >=0 as its num_neurons, you provided {params["num_neurons"]}')
        if not isinstance(params['neuron_class'], str) and not isinstance(params['neuron_class'], dict):
            raise Exception(f'Object NeuronPopulation requires a string or dictionary as its neuron class, you provided {params["neuron_class"]}')
        if isinstance(params['neuron_class'], dict):
            self._validate_custom_neuron_class(params['neuron_class'])

    def __init__(self, params):
        """Builds a neuron object from a dictionary containing all its parameters
        Parameters
        ----------
        params : dict
            The dictionary containing all the parameters necessary to build the neuron
        name : str
            The unique identifier of the neuron population
        """
        self._validate_input(params)
        self.pop_name = params['pop_name']
        self.num_neurons = params['num_neurons']
        self.param_space = params['param_space'] #{'name' : value}
        self.spike_recording_enabled = params['spike_recording_enabled']
        if isinstance(params['neuron_class'], str):
            #Use one of GeNN default neuron classes
            self.neuron_class = params['neuron_class']
        else:
            #Define a custom neuron class
            self.neuron_class = self._init_custom_neuron(params['neuron_class'])

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
        res = network.add_neuron_population(
            pop_name = self.pop_name,
            num_neurons = self.num_neurons,
            neuron = self.neuron_class,
            param_space = self.param_space,
            var_space = self.var_space
        )
        res.spike_recording_enabled = self.spike_rec
        return res

    def __len__(self):
        """Returns the number of neurons in the population
        Return
        ------
        res : int
            The number of neurons in the population
        """
        return self.num_neurons

    def _init_custom_neuron(self, param):
        res = create_custom_neuron_class(
            #Name of the custom neuron
            class_name = param['class_name'],
            #List of strings with parameter names
            param_names = param['param_names'],
            #List of ("name", "precision", ACCESSMODE)
            var_name_types = param['var_name_types'],
            #List of pairs, where the first member is a string with a name of the parameters
            #and the second should be a functor returned by create_dpf_class
            derived_params = param['derived_params'],
            #String containing the code for executing the integration of the model for one
            #time step (so the differential equations solved numerically),
            #variables referenced as $(NAME)
            sim_code = param['sim_code'],
            #Condition for true spike detection
            threshold_condition_code = param['threshold_condition_code'],
            #Code to be run after a true spike is emitted
            reset_code = param['reset_code'],
            #Code block that contains supporting code used in multiple pieces of code like
            #sim code or threshold condition code
            support_code = param['support_code'],
            #List of ("name", "type") of additional parameters
            extra_global_params = param['extra_global_params'],
            #List of ("name", "types", initial_value) of additional local input variables
            additional_input_vars = param['additional_input_vars'],
            #Whether the model require auto-refractory logic to be generated
            is_auto_refractory_required = param['is_auto_refractory_required'],
            #Dictionary with additional attributes and methods of the new class
            custom_body = param['custom_body']
        )
        return res







if __name__ == '__main__':
    import sys
    from reading_parameters import get_parameters
    params = get_parameters(sys.argv[1])
    orn = NeuronPopulation(params['new_neuron_populations']['or'])
    print(orn)
