from copy import copy
from logging import warning
from pathlib import Path
from re import I
from typing import Dict, List, Union, cast

from matplotlib.pyplot import connect
import numpy as np
from pygenn import genn_wrapper
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
    external_g: Optional[np.NDArray]
        In some cases, an external covariance factor may be needed.
    """

    def __init__(self, param, name, source, target):
        self.name = name
        self.param = param
        self.source = source
        self.target = target
        self._param_transform()

    def _complex_parameters(self, param):
        res = ""
        if isinstance(param, list):
            res = "\n".join(param)
        else:
            res = param
        return res

    def _param_transform(self):
        self.param['w_update_model'] = self._set_w_update_model(self.param['w_update_model'])
        self.param['postsyn_model'] = self._set_post_synaptic(self.param['postsyn_model'])
        self.param['connectivity_initialiser'] = self._set_connectivity_initialiser(self.param['connectivity_initialiser'])

    def _set_w_update_model(self, w_update_model):
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
        if not isinstance(postsyn_model, str):
            postsyn_model['apply_input_code'] = self._complex_parameters(postsyn_model['apply_input_code'])
            postsyn_model = genn.create_custom_postsynaptic_class(
                postsyn_model['name'],
                apply_input_code = postsyn_model['apply_input_code']
            )
        return postsyn_model

    def _len_func(self, len_fun):
        if callable(len_fun):
            res = genn.create_cmlf_class(len_fun)()
        else:
            res = len_fun
        return res

    def _get_param_names(self, param_dict):
        return list(param_dict.keys())

    def _set_connectivity_initialiser(self, connectivity_initialiser):
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
