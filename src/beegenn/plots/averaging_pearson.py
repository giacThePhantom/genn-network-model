import sys
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from .data_manager import DataManager
import scipy.cluster.hierarchy as sch
from matplotlib.colors import ListedColormap
from pathlib import Path
import pandas as pd
import scipy.stats as stats


def get_files(simulation_dir):
    files = list(simulation_dir.glob('*.csv'))
    return files

def get_data(files):
    data = {}
    for file in files:
        data[file.stem.replace('_pearson_rank_test', '')] = pd.read_csv(file)
    return data

def comparison_statistics(data):
    temp = data.groupby('pop')['pearson']
    return temp.mean(), temp.std(), temp.max(), temp.min()

def single_statistics(data):
    temp = data.groupby(['pop1', 'pop2'])['pearson']
    return temp.mean(), temp.std(), temp.max(), temp.min()


if __name__ == "__main__":
    root_dir = Path(sys.argv[1])
    comparing_simulations = root_dir / 'comparison_conditions'
    single_simulations = root_dir / 'single_conditions'

    comparing_files = get_files(comparing_simulations)
    single_files = get_files(single_simulations)

    comparing_data = get_data(comparing_files)
    single_data = get_data(single_files)

    comparison_res = pd.DataFrame(columns = ['first', 'second', 'mean', 'std', 'max', 'min'])
    for i in comparing_data:
        statistics = comparison_statistics(comparing_data[i])
        dict_statistics = dict(statistics[0])
        for pop in dict_statistics:
            df_line = {
                    'first' : i.split('_')[0],
                    'second' : i.split('_')[1],
                    'pop' : pop,
                    'mean': statistics[0][pop],
                    'std' : statistics[1][pop],
                    'max' : statistics[2][pop],
                    'min' : statistics[3][pop]
                    }
            comparison_res = comparison_res.append(df_line, ignore_index=True)

    comparison_res.to_csv(root_dir / 'comparison_res.csv', index=False)

    single_res = pd.DataFrame(columns = ['simulation', 'first_pop', 'second_pop', 'mean', 'std', 'max', 'min'])
    for i in single_data:
        statistics = single_statistics(single_data[i])
        dict_statistics = dict(statistics[0])
        for pop1, pop2 in dict_statistics:
            df_line = {
                    'simulation': i,
                    'first_pop' : pop1,
                    'second_pop' : pop2,
                    'mean': statistics[0][pop1][pop2],
                    'std' : statistics[1][pop1][pop2],
                    'max' : statistics[2][pop1][pop2],
                    'min' : statistics[3][pop1][pop2]
                    }
            single_res = single_res.append(df_line, ignore_index=True)

    single_res.to_csv(root_dir / 'single_res.csv', index=False)
