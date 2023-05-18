import numpy as np
import random
from typing import Optional, cast


class Odor:
    """
    Represents the odors for a protocol, building the
    binding and activation rates for the corresponding channel.

    ...

    Attributes
    ----------
    name : str
        The name of the odor
    param : dict
        The parameters necessary to build the odor
    num_glomeruli : int
        The number of glomeruli for which to build the activation
        and binding rates
    homogeneous : Optional[bool]
        Whether the odor has an homogeneous activation rate.
        If false they are sampled from a Gaussian distribution
        If None the user has to provide already-fitted data (see the constructor for details)

    Methods
    -------
    shuffle_binding_rates(shuffle) : None
        Shuffle the binding rates according to permutation
        "shuffle"
    name(self) :
        Getter for name
    get_cuda_rates(self) : np.ndArray
        Returns the binding and activation rates in cuda-friendly
        form
    binding_rates(self) : np.ndArray
        Getter for binding rates
    activation_rates(self) : np.ndArray
        Getter for activation rates

    """

    def __init__(
        self,
        param,
        name,
        num_glomeruli,
        homogeneous: Optional[bool],
    ):
        """
        Builds the odor class with the corresponding binding and
        activation rates.

        If the user does not have pre-fitted data all they have to do is
        provide these binding rates as lists, otherwise they will be generated as random variables.
        """
        self._name = name
        self.param = param
        self.num_glomeruli = num_glomeruli
        self.homogeneous = homogeneous

        self._binding_rates = self._build_binding_rates()
        self._activation_rates = self._build_activation_rates()
        self._unbinding_rates = np.zeros_like(self._binding_rates)
        self._deactivation_rates = np.zeros_like(self._activation_rates)

    def _compute_random_variable(self, var):
        """From a dictionary containing the parameters sample a
        Gaussian distribution
        Parameters
        ----------
        var : dict
            A dictionary containing all the parameters of a random
            distribution

        Returns
        -------
        res : double
            A real number sampled from the distribution
        """

        if isinstance(var, dict):
            while True:
                res = np.random.normal(var["mu"], var["sigma"])
                if res >= var["interval"][0] and res <= var["interval"][1]:
                    var = res
                    break
        else:
            res = var
        return res

    def _build_binding_rate(self, binding_params, i):
        """From the binding parameters builds a binding rate for a glomerulus
        Parameters
        ----------
        binding_params : dict
            A dictionary containing all the parameters necessary to sample the
            distribution
        i : int
            The index of the glomerulus
        Returns
        -------
        binding_profile : double
            The binding rate for glomerulus i
        """

        # Traslate the index according to the midpoint of the distribution
        x = i - binding_params["midpoint"]
        # Set a threshold on the minimum value for the point
        x = np.minimum(np.abs(x), np.abs(x + self.num_glomeruli))
        # Set a threshold on the maximum value for the point
        x = np.minimum(np.abs(x), np.abs(x - self.num_glomeruli))
        # Compute the binding distribution as a Gaussian profile
        binding_profile = np.power(
            10, self._compute_random_variable(binding_params["amplitude"])
        ) * np.exp(
            -np.power(x, 2)
            / (2 * np.power(self._compute_random_variable(binding_params["sigma"]), 2))
        )
        if binding_profile < binding_params["min_thresh"]:
            binding_profile = 0
        return binding_profile

    def _build_activation_rate(self, prev=None):
        """From the activation parameters builds a activation rate for a glomerulus
        Parameters
        ----------
        prev : object
            Whether a previous activation rate was built
        Returns
        -------
        res : double
            The activation rate for a glomerulus
        """

        res = 0

        if self.homogeneous and prev:
            res = prev
        else:
            res = self._compute_random_variable(self.param["activation"])
        return res

    def _check_rates_list(self, param):
        if isinstance(param, list):
            return np.array(param)
        if isinstance(param, float):
            return np.full(self.num_glomeruli, param)

    def _build_binding_rates(self):
        """Builds the binding rates for all the glomeruli"""

        params = self.param["binding"]
        if (returned := self._check_rates_list(params)) is not None:
            return returned

        res = []
        for i in range(self.num_glomeruli):
            res.append(self._build_binding_rate(params, i))
        return np.array(res)

    def _build_unbinding_rates(self):
        return self._check_rates_list(self.param["unbinding"])

    def _build_activation_rates(self):
        """Builds the activation rates for all the glomeruli"""

        if (returned := self._check_rates_list(self.param["activation"])) is not None:
            return returned

        res = []
        res.append(self._build_activation_rate())
        for i in range(1, self.num_glomeruli):
            res.append(self._build_activation_rate(res[0]))
        return np.array(res)

    def _build_deactivation_rates(self):
        return self._check_rates_list(self.param["deactivation"])

    def shuffle_binding_rates(self, shuffle=None):
        """Permutes the binding rates according to a pre-existing shuffle
        if it exists
        Parameters
        ----------
        shuffle : np.ndArray
            The array of indices on how to do a shuffle
        """

        if not shuffle is None:
            self._binding_rates = self._binding_rates[shuffle]
        else:
            random.shuffle(self._binding_rates)

    @property
    def name(self):
        """Getter for name
        Returns
        -------
        name : str
            The name of the odor
        """

        return self._name

    @property
    def binding_rates(self):
        "The binding_rates (also known as k_1). They are not hill-powered yet."

        return self._binding_rates

    @property
    def unbinding_rates(self):
        "The unbinding_rates (also known as k_-1)"

        return self._unbinding_rates

    @property
    def activation_rates(self):
        "The activation rates (also known as k2)"
        return self._activation_rates

    @property
    def deactivation_rates(self):
        "The activation rates (also known as k2)"
        return self._deactivation_rates


"""
if __name__ == "__main__":
    import sys
    from reading_parameters import get_parameters

    param = get_parameters(sys.argv[1])
    temp = Odor(
        param["protocols"]["experiment1"]["odors"]["default"], "iaa", 160, False
    )
    temp1 = Odor(
        param["protocols"]["experiment1"]["odors"]["default"], "iaa", 160, False
    )
    print(temp.binding_rates)
    test = np.arange(160)
    random.shuffle(test)
    temp.shuffle_binding_rates(test)
    temp1.shuffle_binding_rates(test)
    print(temp.binding_rates)
    print(temp.binding_rates)
    print(temp.binding_rates == temp1.binding_rates)
"""
