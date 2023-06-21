import pandas as pd
import sys


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    df = df.dropna()
    df = df[df["pop"] == 'pn']
    df = df[df['p_value'] < 0.05]
    events = pd.unique(list(zip(df['t_start'], df['t_end'])))

    for event in events:
        df_event = df[(df['t_start'] == event[0]) & (df['t_end'] == event[1])]

    print(sys.argv[1].split('/')[-1].split('_')[0], sys.argv[1].split('/')[-1].split('_')[1], f"{df['spearman'].mean()}", f"{df['spearman'].std()}", sep = ',')
