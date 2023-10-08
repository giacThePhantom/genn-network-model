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



def single_statistics(data):
    temp = data.groupby(['sim_name', 'pop'])['mean']
    print(temp)
    return temp.mean(), temp.std(), temp.max(), temp.min()

if __name__ == "__main__":
    correlation_file = sys.argv[1]
    correlation_data = pd.read_csv(correlation_file)
    single_res = pd.DataFrame(columns = ['simulation', 'pop', 'mean', 'std', 'max', 'min'])

    statistics = single_statistics(correlation_data)
    dict_statistics = dict(statistics[0])
    for pop1, pop2 in dict_statistics:
        df_line = {
                'simulation' : pop1,
                'pop' : pop2,
                'mean': statistics[0][pop1][pop2],
                'std' : statistics[1][pop1][pop2],
                'max' : statistics[2][pop1][pop2],
                'min' : statistics[3][pop1][pop2]
                }
        single_res = single_res.append(df_line, ignore_index=True)

    single_res.to_csv("correlation_statistics.csv")
    print(single_res)
