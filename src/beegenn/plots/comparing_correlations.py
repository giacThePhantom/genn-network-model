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

def correlation_dir_per_sim(root_dir, sim_name, nrun):
    res = {}
    raw_data_dir = root_dir / sim_name / "raw_data"

    for nrun in raw_data_dir.iterdir():
        if nrun.is_dir():
            res[str(nrun).split('/')[-1]] = {}
            for filename in (raw_data_dir  / nrun / "correlation_not_clustered").iterdir():
                if filename.is_file():
                    res[str(nrun).split('/')[-1]][str(filename).split('/')[-1]] = np.genfromtxt(str(filename), delimiter=',')

    return res

def compute_spearman_rank_test(sim_1_corr_files, sim_2_corr_files):
    res = pd.DataFrame(columns=['run1', 'run2', 'pop', 't_start', 't_end', 'spearman', 'p_value'])
    for first_run in sim_1_corr_files.keys():
        for second_run in sim_2_corr_files.keys():
            for i in sim_1_corr_files[first_run].keys():
                if i in sim_2_corr_files[second_run]:
                    pop, t_start, t_end = i[:-4].split('_')
                    spearman = stats.spearmanr(sim_1_corr_files[first_run][i].flatten(), sim_2_corr_files[second_run][i].flatten())

                    df_line = {
                            'run1': first_run,
                            'run2': second_run,
                            'pop': pop,
                            't_start': float(t_start),
                            't_end': float(t_end),
                            'spearman': spearman.statistic,
                            'p_value': spearman.pvalue
                            }
                    res = res.append(df_line, ignore_index=True)

    return res


if __name__ == "__main__":
    root_dir = Path(sys.argv[1])
    out_dir = root_dir / "comparison_conditions"
    out_dir.mkdir(exist_ok=True)
    sim_1_name = sys.argv[2]
    sim_2_name = sys.argv[3]
    sim_1_corr_files = correlation_dir_per_sim(root_dir, sim_1_name, sim_2_name)
    sim_2_corr_files = correlation_dir_per_sim(root_dir, sim_2_name, sim_2_name)

    res = compute_spearman_rank_test(sim_1_corr_files, sim_2_corr_files)

    pd.DataFrame.to_csv(res, str(out_dir / f'{sim_1_name}_{sim_2_name}_spearman_rank_test.csv'))
