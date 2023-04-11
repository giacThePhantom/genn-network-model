import json
import numpy as np
from ..protocols import protocol
from .reading_parameters import parse_cli
import sys
from itertools import compress

def primes(n):
    """ Returns  a list of primes < n for n > 2 """

    sieve = bytearray([True]) * (n//2)
    for i in range(3,int(n**0.5)+1,2):
        if sieve[i//2]:
            sieve[i*i//2::i] = bytearray((n-i*i-1)//(2*i)+1)
    return [2,*compress(range(3,n,2), sieve[1:])]

def factorization(n):
    """ Returns a list of the prime factorization of n """

    pf = []
    for p in primes(n):
      if p*p > n : break
      count = 0
      while not n % p:
        n //= p
        count += 1
      if count > 0: pf.append((p, count))
    if n > 1: pf.append((n, 1))
    return pf

def divisors(n):
    """ Returns an unsorted list of the divisors of n """
    divs = [1]
    for p, e in factorization(n):
        divs += [x*p**k for k in range(1,e+1) for x in divs]
    return divs

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


def split_odors(odors):
    """Splits the generated list of odors into n equally sized lists
    of maximum dimension 50
    """

    div = np.array(divisors(len(odors)))
    div = np.max(div[div < 50])
    res = [{}]
    for i in odors:
        res[-1][i] = odors[i]
        if len(res[-1]) >= div and len(res)*div < len(odors):
            res.append({})
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

    odors = generate_odors_changing_amplitude_and_sigma(
            0,
            4.4,
            0.1,
            3,
            10,
            0.1,
            )

    odors = split_odors(odors)

    for (i, odor) in enumerate(odors):
        protocol_template['odors'] = odor

        protocol_template['num_odors'] = len(protocol_template['odors'])

        new_protocol_out_file = protocol_out_file.split('.')
        new_protocol_out_file = '.'.join(new_protocol_out_file[:-1]) + f'_{i}.' + new_protocol_out_file[-1]
        with open(new_protocol_out_file, 'w') as f:
            json.dump(protocol_template, f, indent = 2)

        with open(sim_in_file) as f:
            simulation_template = json.load(f)


        simulation_template['experiment_name'] = '.'.join(new_protocol_out_file.split('/')[-1].split('.')[:-1])

        new_sim_out_file = sim_out_file.split('.')
        new_sim_out_file = '.'.join(new_sim_out_file[:-1]) + f'_{i}.' + new_sim_out_file[-1]
        with open(new_sim_out_file, 'w') as f:
            json.dump(simulation_template, f, indent = 2)
        print('.'.join(new_sim_out_file.split('.')[:-1]).split('/')[-1])
