import json
import numpy as np
import protocols












n_glo = 160 # TODO: add a neuron population and test here
np.random.seed(0)
hill_exp = np.random.uniform(0.95, 1.05, n_glo)
odors = np.random.randn(100, n_glo, 2).astype(np.float32)

protocol = protocols.exp1_protocol()
print(len(protocol))

parameter_arrays = protocols.convert_protocol_to_cuda(protocol, odors, hill_exp)

print(np.shape(parameter_arrays))
print(parameter_arrays[0][1])
