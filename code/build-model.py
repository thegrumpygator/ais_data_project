# %%
import pandas as pd
import os

filepath = "S:/fulls/Hackathon-2019/Data"
filename = "2015_hackathon_v1.1.2.feather"

df = pd.read_feather(os.path.join(filepath, filename))
df = df.head(1000000).copy()

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

superclass = {
    1024: 1, 80: 1, 81: 1, 82: 1, 83: 1, 84: 1, 89: 1, 1019: 2, 36: 2, 37: 2,
    1001: 3, 30: 3, 70: 4, 71: 4, 72: 4, 73: 4, 74: 4, 75: 4, 77: 4, 79: 4,
    1003: 4, 1004: 4}

df['SuperClass'] = (df
                    ["VesselType"]
                    .apply(lambda x: superclass.get(x, 5))
                    )

# Speed: Speed mean, speed median, speed standard deviation,

# Distance: LAT LON

import numpy as np

df["LAT_cd"] = (df
                .groupby(["MMSI", "track_num"])
                ["LAT"]
                .diff()
                .apply(lambda x: x ** 2)
                .fillna(0)
                )

df["LON_cd"] = (df
                .groupby(["MMSI", "track_num"])
                ["LON"]
                .diff()
                .apply(lambda x: x ** 2)
                .fillna(0)
                )

df["distance"] = df["LAT_cd"] + df["LON_cd"]

df["distance"] = df["distance"].apply(np.sqrt)

df["cumulative_distance"] = (df
                             # (df
                             .groupby(["MMSI", "track_num"])
                             ["distance"]
                             .cumsum()
                             )

df_cd = (df
         .groupby(["MMSI", "track_num"])
         ["cumulative_distance"]
         .last()
         .to_frame()
         )


def fillnan511(x):
    if x == 511:
        return np.nan
    else:
        return x

def modheading(x):
    return np.abs(x) % 360


df["Heading"] = df["Heading"].apply(fillnan511)

df["Heading"] = df.groupby(["MMSI", "track_num"])["Heading"].ffill()

df["cumulative_heading"] = (df
                            # (df
                            .groupby(["MMSI", "track_num"])
                            ["Heading"]
                            .diff()
                            .apply(modheading)
                            # .groupby(["MMSI", "track_num"])
                            # .isna()
                            # .sum()
                            )

df["cumulative_heading"] = (df
                            .groupby(["MMSI", "track_num"])
                            ["cumulative_heading"]
                            .cumsum()
                            )

# df["heading_cumulative"] = (df
#     .groupby(["MMSI", "track_num"])
#     ["Heading"]
#     .diff()
#     .apply(modheading)
# )


# .agg({"column": [function1, function2, np.abs]})

# from scipy import stats


df_speed_mean = (df
    .groupby(["MMSI", "track_num"])
    .agg({
    "cumulative_distance": ["last"],
    "cumulative_heading": ["last"],
    "SOG": ["mean", "median", "std"],
    "Status": [lambda x: x.value_counts().index[0]],
    "Length": [lambda x: np.nanmax(x)],
    "Width": [lambda x: np.nanmax(x)],
    "Draft": [lambda x: np.nanmax(x)],
    "SuperClass": [lambda x: x.value_counts().index[0]],
})

)

df_speed_mean

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()


# for index, dfi in df_speed_mean.groupby(["SuperClass"]):

# print(dfi)

import seaborn as sns

sns.set_style('whitegrid')

# df_speed_mean.hist(by="SuperClass")


# %%

# df_speed_mean.melt()

dfs = df_speed_mean

cols = [
    "cumulative_distance",
    "cumulative_heading",
    "SOG_mean",
    "SOG_median",
    "SOG_std",
    "Status",
    "Length",
    "Width",
    "Draft",
    "SuperClass",
]

dfs.columns = cols

dfs[["Length", "Width", "Draft"]] = dfs[["Length", "Width", "Draft"]].fillna(0)

# %%

# Remote python development in Visual Studio Code

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

dfs2 = dfs.reset_index(drop=True).query("SuperClass != 5")[cols[:-1]]

dfscopy = dfs2.copy()

nparray = scaler.fit_transform(dfscopy)

dfscopy = pd.DataFrame(nparray, columns=cols)

dftp = (dfscopy
        .merge(dfs["SuperClass"].reset_index(),
               left_index=True, right_index=True)
        # .reset_index()
        # .drop(columns=["MMSI", "track_num"])
        .melt(id_vars=["SuperClass"])
        )
# %%

def scale(series):
    series = (series - series.mean()) / series.std()
    series = series.fillna(0)
    return series

feature_cols = [
    "cumulative_distance",
    "cumulative_heading",
    "SOG_mean",
    "SOG_median",
    "SOG_std",
    "Status",
    "Length",
    "Width",
    "Draft",
]

target_col = ["SuperClass"]

dfs2 = dfs.copy()

for col in dfs[feature_cols]:
    dfs2[col] = scale(dfs[col])

dfs2[target_col] = dfs[target_col]

# %%
g = sns.FacetGrid(dftp, col="variable", hue="SuperClass", col_wrap=1)

g.map(plt.hist, 'value')