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

def replace_nan_with_mode(matrix):
    # Find the mode of non-NaN elements
    non_nan_values = matrix[~np.isnan(matrix)]
    mean_value = np.mean(non_nan_values)

    # Replace NaN values with the mode
    matrix[np.isnan(matrix)] = mean_value

    return matrix

def correlation_dir_per_sim(root_dir, sim_name, nrun):
    res = {}
    raw_data_dir = root_dir / sim_name / "raw_data"

    for nrun in raw_data_dir.iterdir():
        if nrun.is_dir():
            res[str(nrun).split('/')[-1]] = {}
            for filename in (raw_data_dir  / nrun / "correlation_not_clustered").iterdir():
                if filename.is_file():
                    res[str(nrun).split('/')[-1]][str(filename).split('/')[-1]] = replace_nan_with_mode(np.genfromtxt(str(filename), delimiter=','))

    return res

def compute_spearman_rank_test_two_conditions(sim_1_corr_files, sim_2_corr_files, test):
    res = pd.DataFrame(columns=['run1', 'run2', 'pop', 't_start', 't_end', test, 'p_value'])
    for first_run in sim_1_corr_files.keys():
        for second_run in sim_2_corr_files.keys():
            for i in sim_1_corr_files[first_run].keys():
                if i in sim_2_corr_files[second_run]:
                    pop, t_start, t_end = i[:-4].split('_')
                    if test == 'spearman':
                        test_res = stats.spearmanr(sim_1_corr_files[first_run][i].flatten(), sim_2_corr_files[second_run][i].flatten())
                    elif test == 'pearson':
                        test_res = stats.pearsonr(sim_1_corr_files[first_run][i].flatten(), sim_2_corr_files[second_run][i].flatten())
                    else:
                        raise ValueError(f"Unknown test {test}")

                    df_line = {
                            'run1': first_run,
                            'run2': second_run,
                            'pop': pop,
                            't_start': float(t_start),
                            't_end': float(t_end),
                            test : test_res.statistic,
                            'p_value': test_res.pvalue
                            }
                    res = res.append(df_line, ignore_index=True)

    return res

def compute_spearman_rank_test_between_populations(sim_1_corr_files, sim_name, test):
    res = pd.DataFrame(columns=['run1', 'pop1', 'pop2', 't_start', 't_end', test, 'p_value'])
    for first_run in sim_1_corr_files.keys():
        for i in sim_1_corr_files[first_run].keys():
            for j in sim_1_corr_files[first_run].keys():
                    pop1, t_start1, t_end1 = i[:-4].split('_')
                    pop2, t_start2, t_end2 = j[:-4].split('_')
                    if pop1 != pop2 and t_start1 == t_start2 and t_end1 == t_end2:
                        if test == 'spearman':
                            test_res = stats.spearmanr(sim_1_corr_files[first_run][i].flatten(), sim_1_corr_files[first_run][j].flatten())
                        elif test == 'pearson':
                            test_res = stats.pearsonr(sim_1_corr_files[first_run][i].flatten(), sim_1_corr_files[first_run][j].flatten())
                        else:
                            raise ValueError(f"Unknown test {test}")

                        df_line = {
                                'run1': first_run,
                                'pop1': pop1,
                                'pop2': pop2,
                                't_start': float(t_start1),
                                't_end': float(t_end1),
                                test : test_res.statistic,
                                'p_value': test_res.pvalue
                                }
                        res = res.append(df_line, ignore_index=True)
    return res



if __name__ == "__main__":
    sim_1_name = sys.argv[2]
    sim_2_name = sys.argv[3]
    test = sys.argv[4]
    root_dir = Path(sys.argv[1])
    out_dir = root_dir / f"comparison_conditions_{test}"
    out_dir.mkdir(exist_ok=True)
    comparison_dir = out_dir / "comparison_conditions"
    comparison_dir.mkdir(exist_ok=True)
    single_dir = out_dir / "single_conditions"
    single_dir.mkdir(exist_ok=True)

    sim_1_corr_files = correlation_dir_per_sim(root_dir, sim_1_name, sim_2_name)
    sim_2_corr_files = correlation_dir_per_sim(root_dir, sim_2_name, sim_2_name)

    res = compute_spearman_rank_test_two_conditions(sim_1_corr_files, sim_2_corr_files, test)
    pd.DataFrame.to_csv(res, str(comparison_dir / f'{sim_1_name}_{sim_2_name}_{test}_rank_test.csv'))

    sim_1_res = compute_spearman_rank_test_between_populations(sim_1_corr_files, sim_1_name, test)
    pd.DataFrame.to_csv(sim_1_res, str(single_dir / f'{sim_1_name}_{test}_rank_test.csv'))

    sim_2_res = compute_spearman_rank_test_between_populations(sim_2_corr_files, sim_2_name, test)
    pd.DataFrame.to_csv(sim_2_res, str(single_dir / f'{sim_2_name}_{test}_rank_test.csv'))
