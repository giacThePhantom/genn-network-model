import scipy.stats as sis
import nolds
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from .data_manager import DataManager
import bct

def feature_to_be_averaged_across_timestep(split_sdf_avg, feature):
    res = []
    for window in split_sdf_avg:
        res_for_timestep = []
        for i in range(window.shape[1]):
            res_for_timestep.append(feature(window[:,i]))
        res.append(np.mean(res_for_timestep))
    return np.array(res)

def global_feature(sdf_avg_split, feature):
    res = []
    for window in sdf_avg_split:
        res.append(feature(window))
        res[-1] = np.mean(res[-1])

    return np.array(res)

def connectivity_matrix(sdf_avg_split, threshold = 0.1):
    mat = np.zeros((len(sdf_avg_split), sdf_avg_split[0].shape[0], sdf_avg_split[0].shape[0]))
    for (t, window) in enumerate(sdf_avg_split):
        for i in range(mat.shape[1]):
            for j in range(mat.shape[2]):
                mat[t,i,j] = sis.pearsonr(window[i, :], window[j, :])[0]

    res = np.zeros(mat.shape)
    res[mat >= threshold] = 1
    res[mat < threshold] = 0
    return res

def compute_connectivity_features(sdf_avg_split, connectivity, feature):
    res = []
    for (window, connectivity_window) in zip(sdf_avg_split, connectivity):
        if feature.__name__ != 'norm' and feature.__name__ != 'efficiency_bin' and feature.__name__ != 'modularity_dir':
            res.append(np.sum(feature(connectivity_window)))
        elif feature.__name__ == 'norm':
            res.append(np.mean(feature(window)))
        elif feature.__name__ == 'efficiency_bin':
            res.append(feature(connectivity_window))
        elif feature.__name__ == 'modularity_dir':
            res.append(np.sum(np.sum(feature(connectivity_window))))

    return np.array(res)


def extract_features(pop, t_start, t_end, data_manager, run, features, connectivity_features, show):

    sdf_avg = data_manager.sdf_per_glomerulus_avg(
            pop,
            t_start,
            t_end,
            run
            )

    length_split = 5000
    sdf_avg_split = np.array_split(sdf_avg, sdf_avg.shape[1] / length_split, axis=1)
    feature_values = {}
    for feature in features:
        if feature.__name__ != 'skew' and feature.__name__ != 'kurtosis':
            feature_values[feature.__name__] = feature_to_be_averaged_across_timestep(sdf_avg_split, feature)
        else:
            feature_values[feature.__name__] = global_feature(sdf_avg_split, feature)

    connectivity = connectivity_matrix(sdf_avg_split)

    for feature in connectivity_features:
        feature_values[feature.__name__] = compute_connectivity_features(sdf_avg_split, connectivity, feature)
    return feature_values




if __name__ == "__main__":
    from beegenn.parameters.reading_parameters import parse_cli
    from pathlib import Path
    import pandas as pd

    param = parse_cli()
    data_manager = DataManager(
        param["simulations"]["simulation"],
        param["simulations"]["name"],
        param["neuron_populations"],
        param["synapses"],
    )

    events = pd.read_csv(Path(param['simulations']['simulation']['output_path']) / param['simulations']['name'] / 'events.csv')

    feature_functions = [
            np.std,
            sis.skew,
            sis.kurtosis,
            nolds.sampen,
            nolds.hurst_rs,
            nolds.dfa,
            ]

    connectivity_features = [
            bct.betweenness_bin,
            bct.transitivity_bd,
            bct.degrees_dir,
            bct.efficiency_bin,
            #bct.modularity_dir,
            np.linalg.norm,
            ]


    for t_start in range(60000, int(data_manager.protocol.simulation_time), 120000):
        t_end = t_start + 60000
        for i in range(data_manager.get_nruns()):
            features = extract_features('pn', t_start, t_end, data_manager, str(i), feature_functions, connectivity_features, show = False)
            features['t_start'] = t_start
            features['t_end'] = t_end
            features['run'] = i

            features = pd.DataFrame(features)
            features.to_csv(Path(param['simulations']['simulation']['output_path']) / param['simulations']['name'] / 'features.csv')
