from copy import copy
from logging import warning
from pathlib import Path
from re import I
from typing import Dict, List, Union, cast

from matplotlib.pyplot import connect
import numpy as np
from pygenn import genn_wrapper
from pygenn.genn_model import (GeNNModel, create_custom_weight_update_class, create_custom_postsynaptic_class,
                               create_custom_sparse_connect_init_snippet_class, create_custom_init_var_snippet_class, init_connectivity, init_var)

# Make pylance happy
DataFormat = Union[float, str, List[str], Dict[str, 'DataFormat']]

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

    def __init__(self, name, source, target, data, external_g=None):
        self.name = name
        source, target = name.split("_")
        self.source = source
        self.target = target
        self.external_g = external_g
        self.__dict__.update(self._parse(data, external_g=external_g))

    @staticmethod
    def _parse(data: DataFormat, _parent_key="", external_g=None, root=None):
        # To keep the grammar simple only dicts can contain other dicts
        if root is None:
            root = data

        if isinstance(data, float) or isinstance(data, int):
            return data

        if isinstance(data, dict):
            return Synapse._parse_dict(data, external_g, root)

        if isinstance(data, list):
            return Synapse._parse_list(data, _parent_key)

        if isinstance(data, str):
            return data
    
    @staticmethod
    def _parse_dict(data, external_g, root):
        parsed = {}
        for k, v in data.items():
            parsed[k] = {}
            parsed_v = parsed[k] = Synapse._parse(v, k, external_g=external_g, root=root)

            if isinstance(parsed_v, dict):
                Synapse._subparse_dict(parsed, k, external_g, root)
        return parsed
    
    @staticmethod
    def _subparse_dict(parsed, k, external_g, root):
        parsed_v = parsed[k]
        name = parsed_v.pop("name", None)

        if k == "w_update_model":
            parsed[k] = create_custom_weight_update_class(name, **parsed_v)
        elif k == "postsyn_model":
            parsed[k] = create_custom_postsynaptic_class(name, **parsed_v)
        elif k == "connectivity":
            _type = parsed_v["type"]
            del parsed_v["type"]
            if _type == "CustomSparse":
                conn = create_custom_sparse_connect_init_snippet_class(name, **parsed_v)
                param_space = root.get("param_space", {})
                parsed["connectivity_initialiser"] = init_connectivity(conn, param_space)
                del parsed["connectivity"]
            elif _type == "CustomInit":
                # this will be popped out later
                parsed[k] = create_custom_init_var_snippet_class(name, **parsed_v)

        elif k == "ini":
            wu_var_space = parsed.pop("ini")
            if wu_var_space.pop("requires_initing", False):
                init_param_space = root["param_space"]
                var_connectivity = wu_var_space.pop("connectivity")
                if external_g is not None: 
                    wu_var_space = {k: external_g*v for k, v in wu_var_space.items()}
                else:
                    wu_var_space = {k: init_var(var_connectivity, init_param_space) for k in wu_var_space}
                
            parsed["wu_var_space"] = wu_var_space

    @staticmethod
    def _parse_list(data: list, _parent_key):
        if len(data):
            if _parent_key == "param_names":
                return data
            elif isinstance(data[0], str):
                return "\n".join(data)
            else:
                return tuple(data)
        return []


    def add_to_network(self, model: GeNNModel):
        """
        Add the synapse population to a network
        Parameters
        ---------
        network : pygenn.genn_model.GeNNModel
            The network model to which the neuron population is added, with
            whether the population has to be recorded
        """
        
        params = vars(self).copy()
        del params["name"] # name -> pop_name
        del params["external_g"]
        params.pop("param_space", None) # no longer needed

        #source = model.neuron_populations[self.source]
        target = model.neuron_populations[self.target]

        if target and target.size > 0:
            params.setdefault("postsyn_model", "ExpCond")
            params.setdefault("wu_param_space", {})
            params.setdefault("wu_post_var_space", {})
            params.setdefault("ps_var_space", {})
            params.setdefault("wu_var_space", {})
            params.setdefault("wu_pre_var_space", {})
            params.setdefault("ps_param_space", {})
            params.setdefault("connectivity_initialiser", None)

            model.add_synapse_population(
                self.name,
                delay_steps=genn_wrapper.NO_DELAY,
                **params
            )
        else:
            warning("The target population is not in the model, or its size is 0. Skipping...")
