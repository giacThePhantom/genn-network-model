"""
Make plots. TODO cleanup
"""
import numpy as np
from matplotlib import pyplot as plt

def plot_sdf_over_c(param, exp_name, **kwargs):
    data, protocol = parse_data(
        param,
        exp_name, {
            "orn": ["spikes"]#, "pn": ["spikes"], "ln": ["spikes"]
        }
    )

    plt.rc("font", size=8)
    plt.rcParams['text.usetex'] = True
    n_or = param["neuron_populations"]["or"]["n"]

    if not isinstance(protocol, FirstProtocol):
        raise ValueError("This plot requires FirstProtocol")
    for pop in ["orn"]:
        concentrations = {"iaa": [0], "geo": [0]}
        for odor, ii in concentrations.items():
            max_sdfs = []
            max_concentrations = []
            for i in ii:
                od_active = protocol.events[i]
                # FIXME
                od_inactive = protocol.events[i]
                od_active_tstart = od_active["t_start"]
                od_inactive_tend = od_inactive["t_end"]

                pop_spikes = data[f"{pop}_spikes"]
                pop_spikes_t = pop_spikes[:, 0]
                pop_spikes_ids = pop_spikes[:, 1]

                left_spikes_active_idx = np.searchsorted(pop_spikes_t, od_active_tstart)
                right_spikes_inactive_idx = np.searchsorted(pop_spikes_t, od_inactive_tend, 'right')
                n_pop = param["neuron_populations"][pop]["n"]
                spikes_t = pop_spikes_t[left_spikes_active_idx:right_spikes_inactive_idx]
                spikes_ids = pop_spikes_ids[left_spikes_active_idx:right_spikes_inactive_idx]

                sigma_sdf = 100.0
                dt_sdf = 1.0
                n_pop = param["neuron_populations"][pop]["n"]
                factor = n_pop // n_or

                sdf = make_sdf(spikes_t, spikes_ids, n_pop, dt_sdf, sigma_sdf)
                avgd = glo_avg(sdf, factor)

                max_sdfs.append(np.max(avgd))
                max_concentrations.append(od_active["concentration"])


            print(max_concentrations)
            plt.plot(max_concentrations, max_sdfs, label=odor)
        plt.legend()
        plt.show()

        if len(spikes_t) == 0:
            break

        sigma_sdf = 100.0
        dt_sdf = 1.0

def plot_mono(param, exp_name, **kwargs):
    # Abridged from Prof. Nowotny's notebook:
    # Monotonicity is the difference between the maximal value of the time-averaged
    # SDF response and the value at the maximal concentration.
    # - If mono is high, the meanSDF concentratin curve has a local maximum at a lower concentration.
    # - If mono is zero, then there is a high correlation between meanSDF and

    data, protocol = parse_data(
        param,
        exp_name, {
            "orn": ["spikes"], "pn": ["spikes"], "ln": ["spikes"]
        }
    )

    if not isinstance(protocol, FirstProtocol):
        raise ValueError("This plot requires ThirdProtocol")

    protocol = cast(FirstProtocol, protocol)

    pn_spike_t = data["pn_spikes"][:, 0]
    pn_spike_ids = data["pn_spikes"][:, 1]
    n_pop = param["neuron_populations"]["pn"]["n"]
    n_or = param["neuron_populations"]["or"]["n"]

    sigma_sdf = 100.0
    dt_sdf = 1.0


    # FIXME
    n_odors = 3
    n_conc = 3

    max_activation_per_app = np.array((n_odors, n_conc))
    avg_activation_per_app = np.array((n_odors, n_conc))
    max_glo_per_odor = np.array((n_odors,))

    for odor in range(n_odors):
        total_mean_sdf_per_odor = np.zeros(n_or)
        for conc in range(n_conc):
            cur_step = protocol.events[odor*3 + conc]
            #t_start = cur_step["t_start"]
            #t_end = cur_step["t_end"]
            left_active_idx, _, right_inactive_idx = select_window(pn_spike_t, protocol, i)
            spikes_t = pn_spike_t[left_active_idx:right_inactive_idx]
            spikes_ids = pn_spike_ids[left_active_idx:right_inactive_idx]

            sdf = make_sdf(spikes_t, spikes_ids, n_pop, n_pop, dt_sdf, sigma_sdf)
            factor = n_pop // n_or
            avg_glo_sdf = glo_avg(sdf, factor)
            avg_glo_sdf_onlyactive = avg_glo_sdf[3000:6000, :] # this is nasty
            mean_sdf_per_odor = np.mean(avg_glo_sdf_onlyactive, axis=0)
            max_sdf_per_odor = np.amax(avg_glo_sdf, axis=0)
            total_mean_sdf_per_odor += mean_sdf_per_odor

        '''
        max_glo_per_odor[i] = np.argmax(total_mean_sdf_per_odor)
        for conc in range(n_conc):
            max_activation_per_app[odor, conc] = max_sdf_per_odor
            avg_activation_per_app[odor, conc] = mean_sdf_per_odor
        '''


    max_mono = np.zeros(n_odors)
    mean_mono = np.zeros(n_odors)
    '''
    for i in range(n_odors):
        # ("max activation ever recorded for an odor" - "maximum activation at the maximum concentration")
        # ------------------------------------------------------------------------------------------------
        #                ("average of [maximum activation of each concentration] ")
        max_mono[i] = (np.amax(max_activation_per_app[i, :]) - max_activation_per_app[i, -1]) /
        mean_mono[i] = (np.amax(avg_activation_per_app[i, :]) - max_activation_per_app[i, -1]) /
    '''


if __name__ == "__main__":
    parser = get_argparse_template()
    plot_group = parser.add_argument_group("plot")
    plot_group.add_argument("plot", choices=['spikes', 'heatmap', 'sdf-over-c', 'mono'])
    plot_group.add_argument("--precision", help="For spikes plotting, set the desired resolution (in ms). Defaults to the simulation dt.")
    params = parse_cli(parser)
    name = params['simulation']['name']

    match params["cli"]["plot"]:
        case "sdf-over-c":
            f = plot_sdf_over_c
        case "mono":
            f = plot_mono

    f(params, name, precision=params["cli"]["precision"])
