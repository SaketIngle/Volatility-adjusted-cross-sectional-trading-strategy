"""Create the pre-2020 / post-2020 sub-period files used for sub-period runs. Run from the repository root."""
import pandas as pd
d = pd.read_csv("data/Dataset_PS-A.csv", parse_dates=["date"])
d[d.date <= "2019-12-31"].to_csv("data/pre2020.csv", index=False)
d[d.date >= "2020-01-01"].to_csv("data/post2020.csv", index=False)
print("wrote data/pre2020.csv and data/post2020.csv")
