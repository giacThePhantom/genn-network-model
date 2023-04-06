import json
import numpy as np
from ..protocols import protocol
from .reading_parameters import parse_cli
import sys

def generate_odors_changing_amplitude(start, end, steps):
    res = {}
    for i in np.arange(start, end, steps):
        res[str(i)] = {
                "binding" : {
                    "amplitude" : i,
                    "sigma" : 0.0001,
                    "midpoint" : 0,
                    "min_thresh" : 0
                    },
                "activation": {
                    "mu" : 0.02,
                    "sigma" : 0.02,
                    "interval" : [
                        0.003,
                        0.2
                        ]
                    },
                }
    return res




if __name__ == '__main__':
    protocol_in_file = sys.argv[1]
    protocol_out_file = sys.argv[2]
    sim_in_file = sys.argv[3]
    sim_out_file = sys.argv[4]


    with open(protocol_in_file) as f:
        protocol_template = json.load(f)
    protocol_template['odors'] = {}
    out_name = protocol_out_file

    protocol_template['odors'] = generate_odors_changing_amplitude(0, 5, 0.1)

    protocol_template['num_odors'] = len(protocol_template['odors'])

    with open(protocol_out_file, 'w') as f:
        json.dump(protocol_template, f, indent = 2)

    with open(sim_in_file) as f:
        simulation_template = json.load(f)


    simulation_template['experiment_name'] = '.'.join(protocol_out_file.split('/')[-1].split('.')[:-1])

    with open(sim_out_file, 'w') as f:
        json.dump(simulation_template, f, indent = 2)
