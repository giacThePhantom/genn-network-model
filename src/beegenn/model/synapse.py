import pygenn.genn_model as genn


class Synapse:
    """
    Build a custom synapse population type and define its parameters, so to
    Interface it with a network model later on.

    ...

    Attributes
    ----------
    name : str
        A unique identifier for the synapse object
    source : NeuronPopulation
        The pre-synaptic population
    target : NeuronPopulation
        The post-synaptic population
    param : dict
        All the parameters needed to build the SynapseGroup genn object for
        which this class is a wrapper
    """

    def __init__(self, param, name, source, target):
        """Builds a Synapse object from a dictionary containing all its parameters
        Parameters
        ----------
        param : dict
            A dictionary containing all parameters needed for the creation of the
            genn object
        name : str
            The name of the Synapse population
        source : NeuronPopulation
            The pre-synaptic population
        target : NeuronPopulation
            The post-synaptic population
        """

        self.name = name
        self.param = param
        self.source = source
        self.target = target
        self._param_transform()

    def _complex_parameters(self, param):
        """Gets a parameter in the form of a list of strings and returns the joined
        string, assuming each element to be a line in the result
        Parameters
        ----------
        param : list
            The list of string to be coerced into a string
        Returns
        -------
        res : str
            The string resulting from joining param
        """

        res = ""
        if isinstance(param, list):
            res = "\n".join(param)
        else:
            res = param
        return res

    def _param_transform(self):
        """Performs all the necessary transformations from how they are written in
        the parameters' file into genn-friendly formats
        """

        self.param['w_update_model'] = self._set_w_update_model(self.param['w_update_model'])
        self.param['postsyn_model'] = self._set_post_synaptic(self.param['postsyn_model'])
        self.param['connectivity_initialiser'] = self._set_connectivity_initialiser(self.param['connectivity_initialiser'])

    def _set_w_update_model(self, w_update_model):
        """Transforms the parameters into the w_update_model for creating a synapse
        Parameters
        ----------
        w_update_model : dict
            The parameters of the update model to be transformed into genn format
        """

        if not isinstance(w_update_model, str):
            w_update_model['synapses_dynamics_code'] = self._complex_parameters(w_update_model['synapses_dynamics_code'])
            w_update_model['sim_code'] = self._complex_parameters(w_update_model['sim_code'])
            w_update_model = genn.create_custom_weight_update_class(
                w_update_model['name'],
                param_names = w_update_model['param_names'],
                var_name_types = w_update_model['var_name_types'],
                sim_code = w_update_model['sim_code'],
                synapse_dynamics_code = w_update_model['synapses_dynamics_code']
            )
        return w_update_model

    def _set_post_synaptic(self, postsyn_model):
        """Transforms the parameters into the postsyn_model for creating a synapse
        Parameters
        ----------
        postsyn_model : dict
            The parameters of the post-synaptic model to be transformed into genn format
        """

        if not isinstance(postsyn_model, str):
            postsyn_model['apply_input_code'] = self._complex_parameters(postsyn_model['apply_input_code'])
            postsyn_model = genn.create_custom_postsynaptic_class(
                postsyn_model['name'],
                apply_input_code = postsyn_model['apply_input_code']
            )
        return postsyn_model

    def _len_func(self, len_fun):
        """Transform the function to compute the maximum length of a column or row into
        genn format
        Parameters
        ----------
        len_fun : str or fun
            A function or a string representing the maximum length of a row or column
        """

        if callable(len_fun):
            res = genn.create_cmlf_class(len_fun)()
        else:
            res = len_fun
        return res

    def _get_param_names(self, param_dict):
        """Returns all the parameters name
        Parameters
        ----------
        param_dict : dict
            The dictionary for which the names of parameters are returned
        Returns
        -------
        res : list
            A list of the name of parameters
        """

        return list(param_dict.keys())

    def _set_connectivity_initialiser(self, connectivity_initialiser):
        """Creates the connectivity initialiser to create genn's synapses
        Parameters
        ----------
        connectivity_initialiser : dict
            A dictionary containing all the parameters necessary to create the
            connectivity initialiser
        Returns
        -------
        res : InitSparseConnectivitySnippet
            The connectivity snippet for genn synapses
        """

        if not isinstance(connectivity_initialiser, str) and connectivity_initialiser:
            connectivity_initialiser['row_build_code'] = self._complex_parameters(connectivity_initialiser['row_build_code'])
            connectivity_initialiser['col_build_code'] = self._complex_parameters(connectivity_initialiser['col_build_code'])
            connectivity_initialiser['calc_max_row_len_func'] = self._len_func(connectivity_initialiser['calc_max_row_len_func'])
            connectivity_initialiser['calc_max_col_len_func'] = self._len_func(connectivity_initialiser['calc_max_col_len_func'])
            param_names = self._get_param_names(connectivity_initialiser['param_space'])
            connectivity_snippet = genn.create_custom_sparse_connect_init_snippet_class(
                class_name = connectivity_initialiser['name'],
                param_names = param_names,
                derived_params = connectivity_initialiser['derived_params'],
                row_build_code = connectivity_initialiser['row_build_code'],
                row_build_state_vars = connectivity_initialiser['row_build_state_vars'],
                col_build_code = connectivity_initialiser['col_build_code'],
                calc_max_row_len_func = connectivity_initialiser['calc_max_row_len_func'],
                col_build_state_vars = connectivity_initialiser['col_build_state_vars'],
                calc_max_col_len_func = connectivity_initialiser['calc_max_col_len_func'],
                calc_kernel_size_func = connectivity_initialiser['calc_kernel_size_func'],
                extra_global_params = connectivity_initialiser['extra_global_params'],
                custom_body = connectivity_initialiser['custom_body'],
            )
            connectivity_initialiser = genn.init_connectivity(connectivity_snippet, connectivity_initialiser['param_space'])
        return connectivity_initialiser

    def add_to_network(self, model: genn.GeNNModel):
        """
        Add the synapse population to a network
        Parameters
        ---------
        network : pygenn.genn_model.GeNNModel
            The network model to which the neuron population is added, with
            whether the population has to be recorded
        """

        res = model.add_synapse_population(
            pop_name = self.param['name'],
            matrix_type = self.param['matrix_type'],
            delay_steps = self.param['delay_steps'],
            source = self.source,
            target = self.target,
            w_update_model = self.param['w_update_model'],
            wu_param_space = self.param['wu_param_space'],
            wu_var_space = self.param['wu_var_space'],
            wu_pre_var_space = self.param['wu_pre_var_space'],
            wu_post_var_space = self.param['wu_post_var_space'],
            postsyn_model = self.param['postsyn_model'],
            ps_param_space = self.param['ps_param_space'],
            ps_var_space = self.param['ps_var_space'],
            connectivity_initialiser = self.param['connectivity_initialiser']
          )
        return res
