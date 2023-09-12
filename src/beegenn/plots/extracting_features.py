import scipy.stats as sis
import nolds
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from .data_manager import DataManager
import bct

def feature_to_be_averaged_across_timestep(sdf_avg, feature):
    res_for_timestep = []
    for i in range(sdf_avg.shape[1]):
        res_for_timestep.append(feature(sdf_avg[:,i]))
    return np.mean(res_for_timestep)

def global_feature(sdf_avg, feature):
    res = []
    res = feature(sdf_avg)
    return np.mean(res)

def connectivity_matrix(sdf_avg, threshold = 0.1):
    mat = np.zeros((sdf_avg.shape[0], sdf_avg.shape[0]))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mat[i,j] = sis.pearsonr(sdf_avg[i, :], sdf_avg[j, :])[0]

    res = np.zeros(mat.shape)
    res[mat >= threshold] = 1
    res[mat < threshold] = 0
    return res

def compute_connectivity_features(sdf_avg, connectivity, feature):
    res = None
    if feature.__name__ != 'norm' and feature.__name__ != 'efficiency_bin' and feature.__name__ != 'modularity_dir':
        res = np.sum(feature(connectivity))
    elif feature.__name__ == 'norm':
        res = np.mean(feature(sdf_avg))
    elif feature.__name__ == 'efficiency_bin':
        res = feature(connectivity)
    elif feature.__name__ == 'modularity_dir':
        res = np.sum(np.sum(feature(connectivity)))

    return res


def extract_features(pop, t_start, t_end, data_manager, run, features, connectivity_features, show):

    sdf_avg = data_manager.sdf_per_glomerulus_avg(
            pop,
            t_start,
            t_end,
            run
            )

    feature_values = {}
    for feature in features:
        if feature.__name__ != 'skew' and feature.__name__ != 'kurtosis':
            feature_values[feature.__name__] = feature_to_be_averaged_across_timestep(sdf_avg, feature)
        else:
            feature_values[feature.__name__] = global_feature(sdf_avg, feature)

    connectivity = connectivity_matrix(sdf_avg)

    for feature in connectivity_features:
        feature_values[feature.__name__] = compute_connectivity_features(sdf_avg, connectivity, feature)
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


    all_features = []
    temp = 0
    for t_start in range(0, int(data_manager.protocol.simulation_time), 60000):
        t_end = t_start + 60000
        for i in range(data_manager.get_nruns()):
            temp += 1
            features = extract_features('pn', t_start, t_end, data_manager, str(i), feature_functions, connectivity_features, show = False)
            features['t_start'] = t_start
            features['t_end'] = t_end
            features['run'] = i
            features['simulation'] = param['simulations']['name']
            all_features.append(features)

    features_df = pd.DataFrame(all_features)
    root_dir = Path(param['simulations']['simulation']['output_path']) / "features"
    root_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in root_dir.glob(param['simulations']['name'] + "*.csv"):
        n = max(n, int(str(i).split('/')[-1].split('.')[-2].split('_')[-1]))

    n += 1


    filename = param['simulations']['name'] + f"_features_{n}.csv"
    features_df.to_csv(str(root_dir / filename))
