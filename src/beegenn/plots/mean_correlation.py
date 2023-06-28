
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

def correlation_dir_per_sim(root_dir):

    res = {}
    for sim_name in root_dir.iterdir():
        if (sim_name/"raw_data").is_dir():
            sim_name_str = str(sim_name).split('/')[-1]
            res[sim_name_str] = {}
            raw_data_dir = root_dir / sim_name / "raw_data"

            for nrun in raw_data_dir.iterdir():
                if nrun.is_dir():
                    res[sim_name_str][str(nrun).split('/')[-1]] = {}
                    for filename in (raw_data_dir  / nrun / "correlation_not_clustered").iterdir():
                        if filename.is_file():
                            res[sim_name_str][str(nrun).split('/')[-1]][str(filename).split('/')[-1]] = np.mean(np.genfromtxt(str(filename), delimiter=','))

    return res


if __name__ == "__main__":
    root_dir = Path(sys.argv[1])
    sim_corr_means = correlation_dir_per_sim(root_dir)
    df = pd.DataFrame(columns=['sim_name', 'run', 'pop', 't_start', 't_end', 'mean'])

    for i in sim_corr_means:
        for z in sim_corr_means[i]:
            for j in sim_corr_means[i][z]:
                row = {
                        "sim_name": i,
                        "run" : z,
                        "pop" : j.split('_')[0],
                        "t_start" : j.split('_')[1],
                        "t_end" : j.split('_')[2].split('.')[0],
                        "mean" : sim_corr_means[i][z][j]
                        }
                df = df.append(row, ignore_index=True)

    df.to_csv(str(root_dir / "mean_correlations.csv"))
