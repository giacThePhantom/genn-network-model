import seaborn as sns
import matplotlib as plt
import sys
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], index_col=0)
    sns.set(style="ticks")
    sns.set_palette("deep")
    fig=sns.pairplot(df.iloc[:,:],hue="t_start")
    fig.savefig("pairplot.png")
