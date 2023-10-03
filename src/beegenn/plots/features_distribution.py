import seaborn as sns
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def merge_same_condition(feature_dir):
    res = {}
    for filename in feature_dir.glob("*.csv"):
        condition = filename.stem.split("_")[0]
        if condition not in res:
            res[condition] = []
        res[condition].append(filename)
    return res

def get_dataframe(file_list):
    res = None
    for filename in file_list:
        if res is None:
            res = pd.read_csv(str(filename))
        else:
            res = pd.concat([res, pd.read_csv(str(filename))], ignore_index = True)
        res = res.drop('Unnamed: 0', axis = 1)
        res = res.drop('t_start', axis = 1)
        res = res.drop('t_end', axis = 1)
        res = res.drop('run', axis = 1)
        res['simulation'] = res['simulation'].str.replace('t30noinput', '')
        res['simulation'] = res['simulation'].str.replace('synapsespoissoncluster', '')
        res['simulation'] = res['simulation'].str.replace('poissoncluster', 'normal')
    return res

if __name__ == "__main__":
    feature_dir = Path(sys.argv[1])
    conditions = merge_same_condition(feature_dir)
    feature_distribution = {}
    out_dir = feature_dir / '..' / 'features_evolution'
    out_dir.mkdir(parents = True, exist_ok = True)

    for condition in conditions:
        conditions[condition] = get_dataframe(conditions[condition])

    for condition in conditions:
        feature_distribution[condition] = {}
        for feature in conditions[condition]:
            if feature != 'simulation':
                feature_distribution[condition][feature] = {}
                feature_distribution[condition][feature]['mean'] = conditions[condition][feature].mean()
                feature_distribution[condition][feature]['sdf'] = conditions[condition][feature].std()

    feature_distribution['0'] = feature_distribution.pop('t30noinputpoissoncluster')
    feature_distribution['2'] = feature_distribution.pop('t30noinputhalvedsynapsespoissoncluster')
    feature_distribution['4'] = feature_distribution.pop('t30noinputquartersynapsespoissoncluster')
    feature_distribution['10'] = feature_distribution.pop('t30noinputtenthsynapsespoissoncluster')
    feature_distribution['100'] = feature_distribution.pop('t30noinputhundrethsynapsespoissoncluster')

    for feature in feature_distribution['0']:
        conditions = [float(i) for i in feature_distribution.keys()]
        means = []
        stds = []
        for condition in feature_distribution.keys():
            means.append(feature_distribution[str(condition)][feature]['mean'])
            stds.append(feature_distribution[str(condition)][feature]['sdf'])
        markers, caps, bars = plt.errorbar(conditions, means, stds, fmt = 'o-', markersize = 8, capsize = 5)
        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        plt.title(feature.split('_')[0])
        plt.xlabel('Inhibitory synapses reduction factor')
        plt.savefig(str(out_dir / (feature + "_evolution.png")))
        plt.clf()
        plt.cla()
