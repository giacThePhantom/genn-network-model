import argparse
import logging
from pathlib import Path
import json
import numpy as np
from pygenn import genn_wrapper
# needed for lambda evaluation
from pygenn.genn_model import create_cmlf_class

def get_all_parameter_files(dir_name):
    """From the input directory exports all parameter files in json format.
    Parameters
    ----------
    dir_name : str
        The root directory containing all the json parameter files.
    Return
    ------
    res : list
        List containing all the relative path to the json parameter files.
    """
    par_dir = Path(dir_name)
    return [str(x) for x in par_dir.glob("*/*.json")]

def read_json_data(filename):
    """Reads a json file and returns the corresponding dictionary.
    Parameters
    ----------
    filename : str
        The name of the json file containing the parameters.
    Return
    ------
    res : dict
        A dictionary containing all the parameters extracted from the json file.
    """
    with open(filename, 'r') as data:
        res = json.load(data)
    return res

def add_params_to_dict(in_dict, filename, json_data):
    """Add all the information extracted from a parameter file to a dictionary,
       so that the information from file /a/b/c.json is saved as
       in_dict['a']['b']['c'].
    Parameters
    ----------
    in_dict : dict
        The dictionary to populate with all the parameter files' data.
    filename : str
        The name of the file from which the data was extracted.
    json_data : dict
        The information extracted from the filename file.

    """
    if filename[0] == '/':
        filename = filename[1:]
    depth = filename.split('/') #Extract the file system structure so to insert it with the same hierarchy in the dictionary
    for (i, key) in enumerate(depth):
        if i == len(depth)-1:
            key = key.split('.')[0] #Removing .json suffix
            in_dict.setdefault(key, json_data) #Adding the data to the dictionary
        else:
            in_dict.setdefault(key, {}) #Add hierarchy structure to the dictionary
            in_dict = in_dict[key] #Go down the hierarchy of the dictionary

def get_all_params(dir_name):
    """From the directory extracts all json files and populate a dictionary
    containing all the corresponding parameter's data.
    Parameters
    ----------
    dir_name : str
        The name of the root directory containing the parameter files.
    Return
    ------
    res : dict
        A dictionary containing all the parameters, with an hierarchy resembling the file-system one.

    """
    files = get_all_parameter_files(dir_name) #Extract file from directory
    res = {}
    for filename in files:
        json_data = read_json_data(filename) #Get json data
        add_params_to_dict(res, filename.replace(dir_name, ''), json_data) #Add to resulting dictionary
    return res

def is_evaluable(param_str):
    """Returns whether a string parameter has to be evaluated.
       An evaluable string parameter is defined as such if it begins with the eval
       keyword.
    Parameters
    ----------
    param_str : str
        A parameter in string format.
    Return
    ------
    res : bool
        Whether the parameter has to be evaluated.

    """
    res = param_str.startswith('eval')
    return res



def get_evaluable_strings(params, evaluable_params):
    """From the total dictionary of parameters populates a list containing
       tuples of (collection, index) that have to be evaluated as defined
       in @is_evaluable.
    Parameters
    ----------
    params : dict
        The dictionary containing all the parameters.
    evaluable_params : list
        An empty list which is populated by tuple of (collection, index) of
        parameters that have to be evaluated.
    """
    to_go_down = [params] #This list contains all dictionaries and list found while exploring the parameter dictionary
    while to_go_down: #Explore the dictionary
        param = to_go_down.pop() #Element to be considered
        if isinstance(param, dict):
            #If the element is a dictionary
            for element in param:
                if isinstance(param[element], dict):
                    to_go_down.append(param[element]) #If a dictionary it has to be explored further
                elif isinstance(param[element], list):
                    to_go_down.append(param[element]) #If a list it has to be explored further
                elif isinstance(param[element], str) and is_evaluable(param[element]):
                    evaluable_params.append((param, element)) #If a string and evaluable has to be added to the list as a pair (dict, key)
        elif isinstance(param, list):
        #If it is a list
            for (i, element) in enumerate(param):
                if isinstance(element, dict):
                    to_go_down.append(element) #If a dictionary it has to be explored further
                elif isinstance(element, list):
                    to_go_down.append(element) #If a list it has to be explored further
                elif isinstance(element, str) and is_evaluable(element):
                    evaluable_params.append((param, i)) #If a string and evaluable has to be added to the list as a pair (dict, index)

def sort_params(param):
    """Defines the order in which parameters have to be evaluated.
    Parameters
    ----------
    param : str
        The parameter to be evaluated.

    Return
    ------
    res : int
        The order of evaluation of the parameters.

    """
    res = param[0][param[1]].split()[1][:-1] #In an evaluable parameters eval is followed by a number, which defines the order in which they have to be evaluated
    try:
        int(res)
    except:
        raise Exception(f"Could not read order of evaluation, string should be in the form eval n: expression, beware of whitespaces, in dictionary {param}")
    return(int(res))

def evaluate_param(to_be_eval, params):
    """Evaluates all evaluable parameters.
    Parameters
    ----------
    to_be_eval : list
        List containing all the evaluable parameters as a tuple (collection, element).
    params : dict
        The dictionary containing all the parameters, necessary to include their
        name in the scope of the function
    """
    to_be_eval.sort(key=sort_params)
    for (coll, element) in to_be_eval:
        coll[element] = eval(":".join(coll[element].split(":")[1:]))


def get_parameters(dir_name, sim_name):
    """Return the dictionary containing all the parameters necessary for a simulation.
    Parameters
    ----------
    dir_name : str
        The root directory containing all the parameters' json files.
    sim_name : str
        the name of the simulation (this will create a global key called "simulation/simulation"
        in params that can be referred to in the eval's)
    Return
    ------
    params : dict
        The dictionary containing all the parameters, with a structure resembling
        the file system one.

    """
    params = get_all_params(dir_name)
    to_be_eval = []
    # set the current simulation
    params["simulation"]["simulation"] = params["simulation"][sim_name]
    get_evaluable_strings(params, to_be_eval)
    evaluate_param(to_be_eval, params)
    return params

def parse_cli():
    parser = argparse.ArgumentParser(prog="BeeGenn", description="A comfy wrapper for GeNN")
    parser.add_argument("data", help="The location of the data folder")
    parser.add_argument("sim_name", help="The location of the data folder")
    parser.add_argument("-v", "--verbose", action="count", help="logger verbosity level (max: -vvv)")

    # TODO: add support for custom groups

    args = parser.parse_args()
    level = logging.ERROR
    if args.verbose == 1:
        level = logging.WARNING
    elif args.verbose == 2:
        level = logging.INFO
    elif args.verbose == 3:
        level = logging.DEBUG

    out_path = Path("outputs") / (args.sim_name + ".log")
    out_path.parent.mkdir(exist_ok=True)
    logging.basicConfig(filename=str(out_path), level=level)

    params = get_parameters(str(Path(args.data)), args.sim_name)
    params["simulation"]["name"] = args.sim_name
    params["simulation"]["out_path"] = out_path

    return params

if __name__ == "__main__":
    print(parse_cli())
