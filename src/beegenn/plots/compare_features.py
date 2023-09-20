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

def compare_two_conditions(first, second, root_dir):
    print(first, second)
    sns.set(style = 'ticks')
    sns.set_palette('deep')
    fig = sns.pairplot(pd.concat([first, second], ignore_index = True), hue = 'simulation')
    out_dir = root_dir.parent / 'comparing_features'
    out_dir.mkdir(parents = True, exist_ok = True)
    print(str(out_dir / f'{first.simulation[0]}_{second.simulation[0]}.png'))
    fig.savefig(str(out_dir / f'{first.simulation[0]}_{second.simulation[0]}.png'))

if __name__ == "__main__":
    feature_dir = Path(sys.argv[1])
    conditions = merge_same_condition(feature_dir)

    for condition in conditions:
        conditions[condition] = get_dataframe(conditions[condition])

    for first in conditions:
        for second in conditions:
            if first != second:
                compare_two_conditions(conditions[first], conditions[second], feature_dir)
