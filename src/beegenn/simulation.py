import logging
from beegenn.model.network import NeuronalNetwork
from beegenn.parameters.reading_parameters import parse_cli
from beegenn.protocols.protocol import Protocol
from beegenn.protocols.first_protocol import FirstProtocol
from beegenn.protocols.second_protocol import SecondProtocol
from beegenn.protocols.third_protocol import ThirdProtocol
from beegenn.recorder.recorder import Recorder

import numpy as np

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
        param: dict
    ) -> None:
        """
        Builds a simulation objects that contains the network model and
        a recorder that saves to disks all the variables that will be
        recorded
        """

        self.sim_name = sim_name

        self.model = NeuronalNetwork(
            param['simulations']['name'],
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
        self.recorder.enable_spike_recording(self.model)

    def update_target_pop(self, target_pop, current_events, events):
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

        genn_model = self.model.network
        logging.info(
            f"Starting a simulation for the model {genn_model.model_name} that will run for {self.protocol.simulation_time} ms")

        if not genn_model._built:
            logging.info("Build and load")
            self.model.build_and_load(self.param['batch'])
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
        # FIXME
        total_timesteps = round(self.protocol.simulation_time)

        with logging_redirect_tqdm():
            with tqdm(total=total_timesteps) as pbar:
                while genn_model.t < total_timesteps:
                    logging.debug(f"Time: {genn_model.t}")
                    genn_model.step_time()
                    self.update_target_pop(target_pop, current_events, events)
                    self.recorder.record(self.model, save)
                    if genn_model.t % 1 == 0:
                        pbar.update(1)
        self.recorder.flush(self.model)

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
            protocol = FirstProtocol(protocol_data[experiment_name])
    protocol.add_inhibitory_conductance(
        params['synapses']['ln_pn'], params['neuron_populations']['ln']['n'], params['neuron_populations']['pn']['n'])
    protocol.add_inhibitory_conductance(
        params['synapses']['ln_ln'], params['neuron_populations']['ln']['n'], params['neuron_populations']['ln']['n'])

    return protocol


if __name__ == "__main__":
    params = parse_cli()
    protocol = pick_protocol(params)
    protocol.events = protocol.events[:2]

    sim_params = params['simulations']
    name = sim_params['name']
    sim = Simulator(name, protocol,
                    params)

    sim.run(
        save=sim_params[name]['save']
    )
