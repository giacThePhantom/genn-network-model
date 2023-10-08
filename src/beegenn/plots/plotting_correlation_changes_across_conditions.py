import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def sub_sim_name_with_reduction_factor(df):
    df['simulation'] = df['simulation'].str.replace('t30noinputpoissoncluster', '1')
    df['simulation'] = df['simulation'].str.replace('t30noinputcluster', '1')
    df['simulation'] = df['simulation'].str.replace('t30noinputhalvedsynapsespoissoncluster', '2')
    df['simulation'] = df['simulation'].str.replace('t30noinputquartersynapsespoissoncluster', '4')
    df['simulation'] = df['simulation'].str.replace('t30noinputtenthsynapsespoissoncluster', '10')
    df['simulation'] = df['simulation'].str.replace('t30noinputhundrethsynapsespoissoncluster', '100')
    df['simulation'] = df['simulation'].astype(int)
    return df

def plot_correlation_change(df):
    x = list(df['simulation'].unique())
    x.sort()
    y = []
    yerr = []
    for i in x:
        print(i)
        y.append((df[(df['simulation'] == i) & (df['pop'] == 'orn')]['mean'].values - df[(df['simulation'] == i) & (df['pop'] == 'pn')]['mean'].values).item())
        yerr.append((df[(df['simulation'] == i) & (df['pop'] == 'orn')]['std'].values + df[(df['simulation'] == i) & (df['pop'] == 'pn')]['std'].values).item())
        print(y,  yerr)
    print(x, y, )
    plt.errorbar(x, y, yerr = yerr, fmt = 'o-', capsize = 5, markersize = 8)
    plt.xlabel("Reduction factor")
    plt.ylabel("$Corr_{PN}$")
    plt.title("Change in difference in correlation with different reduction factor on the inhibitory synapses")
    plt.show()
    plt.clf()
    plt.cla()


if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1], index_col = 0)
    print(data)
    data = data[data['simulation'].str.contains("cluster")]
    with_poisson = data[data['simulation'].str.contains("poisson")]
    without_poisson = data[data['simulation'] != 't30noinputpoissoncluster']
    print(without_poisson)
    without_poisson = sub_sim_name_with_reduction_factor(without_poisson)
    print(without_poisson)
    plot_correlation_change(without_poisson)
