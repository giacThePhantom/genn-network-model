import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

def read_csvs(dir):
    csvs = list(dir.glob("pn*.csv"))
    res = {csv.stem : np.abs(np.genfromtxt(csv, delimiter=','))  for csv in csvs}
    return res

def compute_data(corr):
    mask = np.tri(corr.shape[0], k=-1, dtype = bool)
    corr = corr[mask]
    return {'mean' : corr.mean(), "min" : corr.min(), "max" : corr.max(), "std" : corr.std()}

def get_events(events):
    res = {}
    start_time = 60000.0
    end_time = 120000.0
    for i in events['l']:
        for j in events['sigma']:
            for k in events['tau']:
                for z in events['c']:
                    for w in events['amplitude']:
                        res[f"pn_{start_time}_{end_time}"] = {
                            'l' : i,
                            'sigma' : j,
                            'tau' : k,
                            'c' : z,
                            'amplitude' : w,
                            }
                        start_time += 120000.0
                        end_time += 120000.0
    return res

def get_correlation_data_over_var(events, data, var):
    events_same_var = {}
    print(events)
    for i in events:
        print(events[i])
        if events[i][var] not in events_same_var:
            print(events[i][var], i)
            events_same_var[events[i][var]] = [i]
        else:
            print(events[i][var], i)
            events_same_var[events[i][var]].append(i)

    for i in events_same_var:
        mean = 0
        std = 0
        min = 0
        max = 0
        for j in events_same_var[i]:
            mean += data[j]['mean']
            std += data[j]['std']
            min += data[j]['min']
            max += data[j]['max']
        print(events_same_var)
        mean /= len(events_same_var[i])
        std /= len(events_same_var[i])
        min /= len(events_same_var[i])
        max /= len(events_same_var[i])
        events_same_var[i] = {'mean' : mean, 'std' : std, 'min' : min, 'max' : max}
    return events_same_var

def subplot_correlation_change(events, data, var):
    events_same_var = get_correlation_data_over_var(events, data, var)
    x = list(events_same_var.keys())
    x.sort()
    mean = [events_same_var[i]['mean'] for i in x]
    std = [events_same_var[i]['std'] for i in x]
    min = [events_same_var[i]['min'] for i in x]
    max = [events_same_var[i]['max'] for i in x]
    plt.errorbar(x, max, yerr = std, label = "max", fmt='o-', capsize=5, markersize=8)
    plt.errorbar(x, min, yerr = std, label = "min", fmt='o-', capsize=5, markersize=8)
    plt.errorbar(x, mean, yerr = std, label = "mean", fmt='o-', capsize=5, markersize=8)
    plt.legend()
    plt.title(var)
    plt.show()
    plt.clf()
    plt.cla()

def plot_correlation_change(events, data, vars):
    for var in vars:
        subplot_correlation_change(events, data, var)




if __name__ == "__main__":
    dir = Path(sys.argv[1]) / "raw_data" / "0" / "correlation_not_clustered"
    with open(sys.argv[2]) as f:
        events = json.load(f)["poisson_input"]
    print(events)
    events = get_events(events)
    csvs = read_csvs(dir)

    data = {}
    for i in csvs:
        data[i] = compute_data(csvs[i])
    new_events = {}
    for i in data:
        if i in events:
            new_events[i] = events[i]
    events = new_events
    plot_correlation_change(events, data, ["l", "c", "amplitude"])
