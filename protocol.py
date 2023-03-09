from odors import Odor

class Protocol:

    def _create_odor(self, amplitude_mu, amplitude_sigma, amplitude_interval, sigma_mu, sigma_sigma, min_sigma):
        amplitude = 0
        while True:
            amplitude = np.random.normal(mu_sig, sig_sig)
            if amplitude >= amplitude_interval[0] and amplitude <= amplitude_interval[1]:
                break
        sigma = 0
        while True:
            sigma = np.random.normal(sigma_mu, sigma_sigma)
            if sigma >= min_sigma:
                break
        return Odor(
            num_glomeruli = self.num_glomeruli,
            binding_midpoint = 0,
            binding_amplitude = amplitude,
            binding_sigma = sigma,
            min_thresh = self.clip,
            activation_interval = self.activation_interval,
            activation_mu = self.activation_mu,
            activation_sigma = self.activation_sigma,
            homogeneous = self.homogeneous,
        )


    def __init__(self, num_odors, num_glomeruli):
        self.num_odors = num_odors
        self.num_glomeruli = num_glomeruli
