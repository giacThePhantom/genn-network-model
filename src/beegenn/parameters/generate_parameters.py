import json
import numpy as np
import sys

def generate_odors_changing_amplitude_and_sigma(a_start, a_end, a_steps, s_start, s_end, s_steps,):
    """Generates odors with different amplitude and sigma values

    Parameters
    ----------
    a_start : float
        The minimum value for the amplitude
    a_end : float
        The maximum value for the amplitude
    a_steps : float
        How much to increase the amplitude each time
    s_start : float
        The minimum value for the sigma
    s_end : float
        The maximum value for the sigma
    s_steps : float
        How much to increase the sigma each time
    """

    res = {}
    for i in np.arange(a_start, a_end, a_steps):
        for j in np.arange(s_start, s_end, s_steps):
            res[f'a:{i};s:{j}'] = {
                    "binding" : {
                        "amplitude" : float(i),
                        "sigma" : float(j),
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

    odors = generate_odors_changing_amplitude_and_sigma(
            0,
            4.4,
            0.1,
            3,
            10,
            0.1,
            )

    protocol_template['odors'] = odors
    protocol_template['num_odors'] = len(protocol_template['odors'])

    with open(protocol_out_file, 'w') as f:
        json.dump(protocol_template, f, indent = 2)

    with open(sim_in_file) as f:
        simulation_template = json.load(f)

    simulation_template['experiment_name'] = '.'.join(protocol_out_file.split('/')[-1].split('.')[:-1])

    with open(sim_out_file, 'w') as f:
        json.dump(simulation_template, f, indent = 2)
