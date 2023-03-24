from collections import defaultdict, deque
from pathlib import Path
import pickle
from typing import List, Tuple
import logging

from network import NeuronalNetwork
from neuron import NeuronPopulation
from reading_parameters import parse_cli
from odors import Odor
from protocol import Protocol
from first_protocol import FirstProtocol
from second_protocol import SecondProtocol
from third_protocol import ThirdProtocol
from test_protocol import TestFirstProtocol

import numpy as np
import tables
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from recorder import Recorder


class Simulator:
    """
    A Simulator automatically launches a network with a given protocol
    and tracks the required populations.

    Attributes
    ----------
    sim_name: str
        The name of the simulation
    model: NeuronalNetwork
        a neuronal network to simulate and track
    protocol: Protocol
        the protocol for the simulation. This will be dumped.
    recorded_vars: dict
        a dictionary of variables that will be recorded
    param: dict
        configuration root

    Methods
    -------
    track_vars() :
        Tracks a list of variables for a population
    """

    def _reset(self):
        self.recorded_vars = {}
        self.model.reinitialize()

    def __init__(
        self,
        sim_name: str,
        protocol: Protocol,
        param: dict
    ) -> None:
        self.sim_name = sim_name

        self.model = NeuronalNetwork(
            param['simulations']['name'],
            param['neuron_populations'],
            param['synapses'],
            param['simulations']['simulation']['dt'],
            optimizeCode=params['simulations']['simulation']['optimize_code'],
            generateEmptyStatePush=params['simulations']['simulation']['generate_empty_state_push']
        )
        self.protocol = protocol
        self.param = param['simulations']['simulation']

        # N_timesteps_to_pull_var is the sampling frequency (in timesteps) for normal (non-event) vars.
        # For example, n_timesteps_to_pull_var=100 means every 100 steps (or 100*dt ms) we pull a variable.
        #

        self.recorder = Recorder(self.param['output_path'],
                                 self.sim_name,
                                 self.param['tracked_variables'],
                                 self.model.connected_neurons,
                                 self.param['batch'],
                                 self.param['n_timesteps_to_pull_var'],
                                 self.param['dt'],
                                 protocol.simulation_time)

        self.recorder.dump_protocol(self.protocol)
        self.recorder.enable_spike_recording(self.model)

    def update_target_pop(self, target_pop, current_events, events):
        for (i, event) in enumerate(current_events):
            if self.model.network.t >= event['t_start'] and not event['happened']:
                event['happened'] = True
                target_pop.vars["kp1cn_" +
                                str(event['channel'])].view[:] = event['binding_rates']
                target_pop.vars["kp2_" + str(event['channel'])
                                ].view[:] = event['activation_rates']
                self.model.network.push_state_to_device("or")

            elif self.model.network.t == event['t_end']:
                target_pop.vars["kp1cn_" + str(event['channel'])].view[:] = np.zeros(
                    np.shape(event['activation_rates']))

                if events[i]:
                    current_events[i] = events[i].pop(0)



    def run(self, poll_spike_readings=False, save=True):
        """
        Run a simulation. The user is advised to call `track_vars` first
        to register which variables to log during the simulation

        Parameters
        ----------

        poll_spike_readings: bool
            if False (default), use the internal SpikeRecorder class to record spike events.
            This is much faster than polling the internal state, but is limited to the internal implementation.
            Otherwise, use the (old) spike event polling method. This means almost all events will be lost between
            readings, however it provides useful "snapshot" views for debugging.
        """
        genn_model = self.model.network
        logging.info(
            f"Starting a simulation for the model {genn_model.model_name} that will run for {self.protocol.simulation_time} ms")

        if not genn_model._built:
            logging.info("Build and load")
            self.model.build_and_load(round(self.param['batch'] / self.param['n_timesteps_to_pull_var']))
            logging.info("Done")
        else:
            logging.info("Reinitializing")
            self.model.reinitialise()

        events = self.protocol.get_events_for_channel()
        current_events = []
        for i in events:
            if i:
                current_events.append(i.pop(0))

        target_pop = self.model.connected_neurons['or']

        # Kickstart the simulation
        total_timesteps = round(self.protocol.simulation_time)

        with logging_redirect_tqdm():
            with tqdm(total=total_timesteps) as pbar:
                while genn_model.t < self.protocol.simulation_time:
                    logging.debug(f"Time: {genn_model.t}")
                    genn_model.step_time()
                    self.update_target_pop(target_pop, current_events, events)
                    self.recorder.record(self.model, save)
                    if genn_model.t % 1 == 0:
                        pbar.update(1)
        self.recorder.flush()

def pick_protocol(params):
    # Pick the correct protocol for the experiment
    protocol_data = params["protocols"]
    match params["simulations"]["simulation"]["experiment_name"]:
        case "experiment1":
            protocol = FirstProtocol(protocol_data["experiment1"])
        case "experiment2":
            protocol = SecondProtocol(protocol_data["experiment2"])
        case "experiment3":
            protocol = ThirdProtocol(protocol_data["experiment3"])
        case "testexperiment":
            protocol = TestFirstProtocol(protocol_data["experiment1"])
        case _:
            protocol = FirstProtocol(protocol_data["experiment1_test"])

    protocol.add_inhibitory_conductance(
        params['synapses']['ln_pn'], params['neuron_populations']['ln']['n'], params['neuron_populations']['pn']['n'])
    protocol.add_inhibitory_conductance(
        params['synapses']['ln_ln'], params['neuron_populations']['ln']['n'], params['neuron_populations']['ln']['n'])

    return protocol


if __name__ == "__main__":
    params = parse_cli()
    protocol = pick_protocol(params)

    sim_params = params['simulations']
    name = sim_params['name']
    sim = Simulator(name, protocol,
                    params)

    sim.run(
        poll_spike_readings=sim_params[name]['poll_spike_readings'],
        save=True
    )
