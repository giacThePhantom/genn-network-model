import numpy as np

class Odor:

    def __init__(self, param, num_glomeruli, homogeneous):
        self.param = param
        self.num_glomeruli = num_glomeruli
        self.homogeneous = homogeneous
        self.binding_rates = self._build_binding_rates()
        self.activation_rates = self._build_activation_rates()

    def _binding_amplitude(self):
        res = 0
        if isinstance(params['binding_amplitude'], dict):
            pass
        else:
            res = params['binding_amplitude']
        return res

    def _build_binding_rates(self):
        res = []
        for i in range(self.num_glomeruli):
            x = i - self.param['binding_midpoint']
            x = np.minimum(np.abs(x), np.abs(x + self.num_glomeruli))
            x = np.minimum(np.abs(x), np.abs(x - self.num_glomeruli))
            binding_profile = np.power(10, self.binding_amplitude)*np.exp(-np.power(x, 2)/2*np.power(self.binding_sigma, 2))
            if binding_profile < self.min_thresh:
                binding_profile = 0
            res.append(binding_profile)
        return res

    def _build_activation_rates(self, homogeneous):
        res = []
        for i in range(self.num_glomeruli):
            x = 0
            if i > 0 and homogeneous:
                x = res[0]
            else:
                print(x < self.activation_interval[0] or x > self.activation_interval[1])
                while True:
                    x = np.random.normal(self.activation_mu, self.activation_sigma)
                    if x >= self.activation_interval[0] and x <= self.activation_interval[1]:
                        break
            res.append(x)
        return res





if __name__ == "__main__":
    temp = Odor(160, 50, 0.1, 0, 0, (0, 4), 1, 0.1, False)
    print(temp.activation_rate)
