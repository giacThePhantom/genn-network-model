import logging
from beegenn.model.network import NeuronalNetwork
from beegenn.parameters.reading_parameters import parse_cli
from beegenn.protocols.protocol import Protocol
from beegenn.protocols.first_protocol import FirstProtocol
from beegenn.protocols.second_protocol import SecondProtocol
from beegenn.protocols.third_protocol import ThirdProtocol
from beegenn.protocols.no_input import NoInput
from beegenn.recorder.recorder import Recorder
from pygenn.genn_model import GeNNModel

import numpy as np
from scipy.signal import convolve
from copy import deepcopy

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


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
        """
        Deletes the collections of recorded variables and
        reinitializes the model
        """

        self.recorded_vars = {}
        self.model.reinitialize()

    def __init__(
        self,
        sim_name: str,
        protocol: Protocol,
        param: dict,
        n_run
    ) -> None:
        """
        Builds a simulation objects that contains the network model and
        a recorder that saves to disks all the variables that will be
        recorded
        """

        self.sim_name = sim_name
        self.run_number = n_run

        self.model = NeuronalNetwork(
            param['simulations']['name'] + "_" + str(n_run),
            param['neuron_populations'],
            param['synapses'],
            param['simulations']['simulation']['dt'],
            optimizeCode=params['simulations']['simulation']['optimize_code'],
            generateEmptyStatePush=params['simulations']['simulation']['generate_empty_state_push'],
            backend = params['simulations']['simulation'].get("backend", None)
        )
        self.protocol = protocol
        self.param = param['simulations']['simulation']

        self.recorder = Recorder(self.param['output_path'],
                                 self.sim_name,
                                 self.param['tracked_variables'],
                                 self.model.connected_neurons,
                                 self.param['batch'],
                                 self.param['n_timesteps_to_pull_var'],
                                 self.param['dt'],
                                 protocol.simulation_time)

        self.recorder.dump_protocol(self.protocol)

    def update_target_pop(self, target_pop, current_events, events, poi_input):
        """
        Updates the target population based on a list of events determined by the protocol

        Parameters
        ----------
        target_pop : NeuronPopulation
            The population for which to update the variables
        current_events : list
            The list of events that are happening right now
        events : list
            The list of all the remaining events
        """

        if poi_input is not None:
            self.model.network.pull_state_from_device("or")
            target_pop.vars['ra'].view[:] = target_pop.vars['ra'].view[:] + poi_input
            self.model.network.push_state_to_device("or")

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
                self.model.network.push_state_to_device("or")

                if events[i]:
                    current_events[i] = events[i].pop(0)




    def run(self, save=True):
        """Run a simulation. The user is advised to call `track_vars` first
        to register which variables to log during the simulation

        Parameters
        ----------

        save : bool
            Whether the recorder has to save the recorded variables to disk
        """

        self.recorder.update_run_number(self.run_number)
        self.recorder.enable_spike_recording(self.model)

        genn_model = self.model.network
        logging.info(
            f"Starting a simulation for the model {genn_model.model_name} that will run for {self.protocol.simulation_time} ms")

        if not genn_model._built:
            logging.info("Build and load")
            self.model.build_and_load(self.param['batch'])
            logging.info("Done")
        else:
            logging.info("Reinitializing")
            self.model.reinitialize()

        events = self.protocol.get_events_for_channel()
        current_events = []
        for i in events:
            if i:
                current_events.append(i.pop(0))

        target_pop = self.model.connected_neurons['or']

        # Kickstart the simulation
        total_timesteps = round(self.protocol.simulation_time)





        poi_input = self.poisson_input(
                l = 0.1,
                sigma = 10,
                tau = 3,
                c = 0.3,
                )




        genn_model.t = 0
        with logging_redirect_tqdm():
            with tqdm(total=total_timesteps, disable = "cluster" in self.sim_name) as pbar:
                while genn_model.t < self.protocol.simulation_time:
                    logging.debug(f"Time: {genn_model.t}")
                    genn_model.step_time()
                    if poi_input is not None:
                        self.update_target_pop(target_pop, current_events, events, poi_input[int(genn_model.t/self.param['dt'])])
                    else:
                        self.update_target_pop(target_pop, current_events, events, None)
                    self.recorder.record(self.model, save)
                    if genn_model.t % 1 == 0:
                        pbar.update(1)
        self.recorder.flush()


    def poisson_process(self, sim_time, dt, l = 0.1):
        poi = np.zeros(int(sim_time / dt) + 1)
        tau = -(1/l) * np.log(l * np.random.rand())
        for i in np.arange(0, sim_time, dt):
            if i <= tau and i + dt > tau:
                poi[int(i * dt)] = 0.5
                tau -= (1/l) * np.log(l * np.random.rand())
        return poi

    def add_template(self, poi, template, c):
        for i in range(len(poi)):
            if poi[i] != 0:
                prob = np.random.rand()
                if prob < c:
                    poi[i] = 0
            if template[i] != 0:
                prob = np.random.rand()
                if prob < c:
                    poi[i] = template[i]
        return poi



    def kernel(self, sigma, tau, dt):
        kernel = np.arange(0, sigma, dt)
        kernel = (1/tau)*np.exp(-kernel/tau)
        return kernel

    def poisson_input(self, l = 0.1, sigma = 5, tau = 2, c = 0.1):
        if self.param['poisson_input']:
            template = self.poisson_process(self.protocol.simulation_time, self.param['dt'], l)
            pois = [self.poisson_process(self.protocol.simulation_time, self.param['dt'], l) for _ in range(160)]
            ker = self.kernel(sigma, tau, self.param['dt'])
            pois = [self.add_template(pois[i], template, c) for i in range(len(pois))]
            pois = [convolve(poi, ker, mode = 'same') for poi in pois]
            return np.array(pois).T
        else:
            return None

def pick_protocol(params):
    """Pick the correct protocol for the experiment

    Parameters
    ----------
    params : dict
        A dictionary containing all the parameters needed for the simulation

    Returns
    -------
    protocol : Protocol
        The protocol object that will be used to generate the events for
        the simulation
    """

    protocol_data = params["protocols"]
    experiment_name = params['simulations']['simulation']['experiment_name']
    match params["simulations"]["simulation"]["experiment_type"]:
        case "first_protocol":
            protocol = FirstProtocol(protocol_data[experiment_name])
        case "second_protocol":
            protocol = SecondProtocol(protocol_data[experiment_name])
        case "third_protocol":
            protocol = ThirdProtocol(protocol_data[experiment_name])
        case _:
            protocol = NoInput(protocol_data[experiment_name])
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
    for i in range(sim_params['simulation']['n_runs']):
        sim = Simulator(name,
                        protocol,
                        deepcopy(params),
                        i,
                        )
        sim.run(
            save=sim_params[name]['save']
        )
