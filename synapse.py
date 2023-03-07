import pygenn.genn_model as genn


class Synapse:
    """
    Build a custom synapse population type and define its parameters, so to
    Interface it with a network model later on

    ...

    Attributes
    ----------
    name : str
        A unique identifier for the synapse object
    matrix_type : str
        How the synapse matrix is stored in memory
    delay_steps : int
        Delay in number of steps
    source : NeuronPopulation
        The pre-synaptic population
    target : NeuronPopulation
        The post-synaptic population
    w_update_model : pyGeNN::WeightUpdateModels
        Weight update model class
    wu_param_space : {}
        Parameters for the weight update class
    wu_var_space : {}
        Initial value for the weight update class
    wu_pre_var_space : {}
        Initial values for the pre synaptic variables
    wu_post_var_space : {}
        Initial values for the post-synaptic variables
    postsyn_model : {}
        Postsynaptic model
    ps_param_space : {}
        Parameters for the postsynaptic model
    ps_var_space : {}
        Initial value for the variables in the postsynaptic model
    connectivity_initialiser: {}
        Define connectivity

    Methods
    -------
    """

    name = ""
    matrix_type = ""
    delay_steps = 0
    source = None
    target = None
    w_update_model = None
    wu_param_space = None
    wu_var_space = None
    wu_pre_var_space = None
    wu_post_var_space = None
    postsyn_model = None
    ps_param_space = None
    ps_var_space = None
    connectivity_initialiser = None
    connectivity = None
    n = 0
    _w_update_model = {}
    _postsyn_model = {}
    _connectivity_initialiser = {}

    def _reset(self):
        self.name = ""
        self.matrix_type = ""
        self.delay_steps = 0
        self.source = None
        self.target = None
        self.w_update_model = None
        self.wu_param_space = None
        self.wu_var_space = {}
        self.wu_pre_var_space = None
        self.wu_post_var_space = None
        self.postsyn_model = None
        self.ps_param_space = None
        self.ps_var_space = None
        self.connectivity_initialiser = None
        self.connectivity = None
        self.n = 0
        self._w_update_model = {}
        self._postsyn_model = {}
        self._connectivity_initialiser = {}

    def _build_update_model(self, w_update_model):
        self._reset()
        if isinstance(w_update_model, dict):
            w_update_model['synapses_dynamics_code'] = "\n".join(w_update_model['synapses_dynamics_code'])
            w_update_model['sim_code'] = "\n".join(w_update_model['sim_code'])


            self._w_update_model = w_update_model
            self.w_update_model = genn.create_custom_weight_update_class(
                self._w_update_model['name'],
                param_names = self._w_update_model['param_names'],
                var_name_types = self._w_update_model['var_name_types'],
                sim_code = self._w_update_model['sim_code'],
                synapse_dynamics_code = self._w_update_model['synapses_dynamics_code']
            )
        else:
            self._w_update_model = w_update_model
            self.w_update_model = w_update_model

    def _build_param_space(self, wu_param_space):
        self.wu_param_space = wu_param_space

    def _build_var_space(self, wu_var_space):
        self.wu_var_space = wu_var_space

    def _build_pre_var_space(self, wu_pre_var_space):
        self.wu_pre_var_space = wu_pre_var_space

    def _build_post_var_space(self, wu_post_var_space):
        self.wu_post_var_space = wu_post_var_space

    def _build_postsyn_model(self, postsyn_model):
        if isinstance(postsyn_model, dict):
            postsyn_model['apply_input_code'] = "\n".join(postsyn_model['apply_input_code'])
            self._postsyn_model = postsyn_model
            self.postsyn_model = genn.create_custom_postsynaptic_class(
                self._postsyn_model['name'],
                self._postsyn_model['apply_input_code']
            )
        else:
            self._postsyn_model = postsyn_model
            self.postsyn_model = postsyn_model

    def _build_param_space(self, ps_param_space):
        self.ps_param_space = ps_param_space

    def _build_var_space(self, ps_var_space):
        self.ps_var_space = ps_var_space

    def _build_connectivity_initialiser(self, connectivity_initialiser):
        if connectivity_initialiser:
            connectivity_initialiser['param_names'] = list(connectivity_initialiser['param_space'].keys())
            if connectivity_initialiser['row_build_code']:
                connectivity_initialiser['row_build_code'] = "\n".join(connectivity_initialiser['row_build_code'])
            if connectivity_initialiser['col_build_code']:
                connectivity_initialiser['col_build_code'] = "\n".join(connectivity_initialiser['col_build_code'])
            if connectivity_initialiser['row_build_state_vars']:
                for i in connectivity_initialiser['row_build_state_vars']:
                    i = tuple(i)
                connectivity_initialiser['row_build_state_vars'] = set(connectivity_initialiser['row_build_state_vars'])
            if connectivity_initialiser['col_build_state_vars']:
                for (i, element) in enumerate(connectivity_initialiser['col_build_state_vars']):
                    connectivity_initialiser['col_build_state_vars'][i] = tuple(element)
                connectivity_initialiser['col_build_state_vars'] = set(connectivity_initialiser['col_build_state_vars'])
            self._connectivity_initialiser = connectivity_initialiser
            self.connectivity_initialiser = genn.create_custom_sparse_connect_init_snippet_class 	(
                class_name = self._connectivity_initialiser['name'],
                param_names = self._connectivity_initialiser['param_names'],
                derived_params =self._connectivity_initialiser['derived_params'],
                row_build_code = self._connectivity_initialiser['row_build_code'],
                row_build_state_vars = self._connectivity_initialiser['row_build_state_vars'],
                col_build_code = self._connectivity_initialiser['col_build_code'],
                col_build_state_vars = self._connectivity_initialiser['col_build_state_vars'],
                calc_max_row_len_func = self._connectivity_initialiser['calc_max_row_len_func'],
                calc_max_col_len_func = self._connectivity_initialiser['calc_max_col_len_func'],
                calc_kernel_size_func = self._connectivity_initialiser['calc_kernel_size_func'],
                extra_global_params = self._connectivity_initialiser['extra_global_params'],
                custom_body = self._connectivity_initialiser['custom_body']
            )


    def _build_connectivity(self):
        if self._connectivity_initialiser:
            self.connectivity = genn.init_connectivity(
                self.connectivity_initialiser,
                self._connectivity_initialiser['param_space']
            )


    def __init__(self, params, name, source, target):
        """Builds a synapse population object from a dictionary containing
           all of its parameters
        Parameters
        ----------
        params : dict
            The dictionary containing all the parameters necessary to
            build the synapse
        name : str
            The unique identifier of the synapse population
        """
        self.name = name
        self.matrix_type = params['matrix_type']
        self.delay_steps = params['delay_steps']
        self.source = source
        self.target = target
        self.n = params['n']
        self._build_update_model(params['w_update_model'])
        self._build_param_space(params['wu_param_space'])
        self._build_var_space(params['wu_var_space'])
        self._build_pre_var_space(params['wu_pre_var_space'])
        self._build_post_var_space(params['wu_post_var_space'])
        self._build_postsyn_model(params['postsyn_model'])
        self._build_param_space(params['ps_param_space'])
        self._build_var_space(params['ps_var_space'])
        self._build_connectivity_initialiser(params['connectivity_initialiser'])
        self._build_connectivity()

    def add_to_network(self, network):
        """Add the synapse population to a network
        Parameters
        ---------
        network : pygenn.genn_model.GeNNModel
            The network model to which the neuron population is added, with
            whether the population has to be recorded
        res : list
            A list containing all the values to create a synapse population for genn
        """
        print(self.source)
        res = network.add_synapse_population(self.name,
                                             self.matrix_type,
                                             self.delay_steps,
                                             self.source,
                                             self.target,
                                             self.w_update_model,
                                             self.wu_param_space,
                                             self.wu_var_space,
                                             self.wu_pre_var_space,
                                             self.wu_post_var_space,
                                             self.postsyn_model,
                                             self.ps_param_space,
                                             self.ps_var_space,
                                             self.connectivity
                                            )
        return res




if __name__ == '__main__':
    import sys
    from reading_parameters import get_parameters
    import neuron
    params = get_parameters(sys.argv[1])
    neur_pop = []
    for i in params['neuron_populations']:
        neur_pop.append(neuron.NeuronPopulation(params['neuron_populations'][i], i))

    for i in neur_pop:
        for j in neur_pop:
            if i.name+'_'+j.name in params['synapses']:
                print(i.name+'_'+j.name)
                syn = Synapse(params['synapses'][i.name+'_'+j.name], i.name+'_'+j.name, i, j)
                print(syn)
