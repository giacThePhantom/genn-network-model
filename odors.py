import numpy as np
import random

class Odor:

    def __init__(self, param, name, num_glomeruli, homogeneous):
        self. name = name
        self.param = param
        self.num_glomeruli = num_glomeruli
        self.homogeneous = homogeneous
        self.binding_rates = self._build_binding_rates()
        self.activation_rates = self._build_activation_rates()

    def _compute_random_variable(self, var):
        if isinstance(var, dict):
            while True:
                res = np.random.normal(var['mu'], var['sigma'])
                if res >= var['interval'][0] and res <= var['interval'][1]:
                    break
        else:
            res = var
        return res

    def _build_binding_rate(self, binding_params, i):
        x = i - binding_params['midpoint']
        x = np.minimum(np.abs(x), np.abs(x + self.num_glomeruli))
        x = np.minimum(np.abs(x), np.abs(x - self.num_glomeruli))
        binding_profile = np.power(10, self._compute_random_variable(binding_params['amplitude']))*np.exp(-np.power(x, 2)/2*np.power(self._compute_random_variable(binding_params['sigma']), 2))
        if binding_profile < binding_params['min_thresh']:
            binding_profile = 0
        return binding_profile

    def _build_activation_rate(self, prev = None):
        res = 0
        if self.homogeneous and prev:
            res = prev
        else:
            res = self._compute_random_variable(self.param['activation'])
        return res


    def _build_binding_rates(self):
        res = []
        for i in range(self.num_glomeruli):
            res.append(self._build_binding_rate(self.param['binding'], i))
        return np.array(res)

    def _build_activation_rates(self):
        res = []
        res.append(self._build_activation_rate())
        for i in range(1, self.num_glomeruli):
            res.append(self._build_activation_rate(res[0]))
        return np.array(res)

    def shuffle_binding_rates(self, shuffle = None):
        if not shuffle is None:
            self.binding_rates = self.binding_rates[shuffle]
        else:
            random.shuffle(self.binding_rates)

    def get_name(self):
        return self.name
    
    def get_cuda_rates(self):
        """Get a cuda-friendly representation of these odors"""
        return np.vstack([self.binding_rates, self.activation_rates]).T

    def get_binding_rates(self):
        return self.binding_rates

    def get_activation_rates(self):
        return self.activation_rates





if __name__ == "__main__":
    import sys
    from reading_parameters import get_parameters
    param = get_parameters(sys.argv[1])
    temp = Odor(param['protocols']['experiment1']['odors']['default'], 'iaa', 160, False)
    temp1 = Odor(param['protocols']['experiment1']['odors']['default'], 'iaa', 160, False)
    print(temp.binding_rates)
    test = np.arange(160)
    random.shuffle(test)
    temp.shuffle_binding_rates(test)
    temp1.shuffle_binding_rates(test)
    print(temp.binding_rates)
    print(temp.binding_rates)
    print(temp.binding_rates == temp1.binding_rates)
