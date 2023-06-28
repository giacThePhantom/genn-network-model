import sys
import pandas as pd
from pathlib import Path



def read_data(filename):
    df = pd.read_csv(filename)
    df = df[df["p_value"] < 0.05]
    df = df[df["pop"] == "pn"]
    return df

def get_condition_name(filename):
    return filename.name.split(".")[0].split("_")[0], filename.name.split(".")[0].split("_")[1]

def build_res_row(df, first, second, test):
    return {
            "first" : first,
            "second" : second,
            "max" : df[test].max(),
            "min" : df[test].min(),
            "mean" : df[test].mean(),
            "std" : df[test].std(),
            }



if __name__ == "__main__":
    root_dir = Path(sys.argv[1])
    test = sys.argv[2]
    test_dir = root_dir / f"comparison_conditions_{test}"

    out = root_dir / f"comparison_conditions_summary_{test}.csv"

    res = pd.DataFrame(columns=["first", "second", "max", "min", "mean", "std"])

    for i in test_dir.iterdir():
        df = read_data(i)
        first, second = get_condition_name(i)
        res_row = build_res_row(df, first, second, test)
        res = res.append(res_row, ignore_index=True)

    res.to_csv(out, index=False)
