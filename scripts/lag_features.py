import pandas as pd



def make_lag_features(df_input, train_orig, lags=[1,2,3,7,12]):
    series = df_input["is_nonzero"]

    df = pd.DataFrame({"y": series})

    # ---- lags ----
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)

    # ---- rolling ----
    df["rolling_mean_3"]  = series.shift(1).rolling(3).mean()
    df["rolling_mean_7"]  = series.shift(1).rolling(7).mean()
    df["rolling_mean_14"] = series.shift(1).rolling(14).mean()

    # ---- time ----
    idx = pd.to_datetime(series.index)
    df["is_thursday"] = (idx.dayofweek == 3).astype(int)
    df["is_weekend"]  = (idx.dayofweek >= 5).astype(int)
    df["is_q4"]       = (idx.quarter == 4).astype(int)

    # ---- category ----
    df["is_tv_video"] = df_input["main_category"].str.lower().isin(
        ["tv", "video", "tv & video"]
    ).astype(int)

    # ---- train-based stats ----
    cat_spike_rate = train_orig.groupby("main_category")["is_nonzero"].mean()
    df["cat_spike_rate"] = df_input["main_category"].map(cat_spike_rate)

    # ---- spike logic ----
    df["prev_2_were_zero"] = ((df["lag_1"] == 0) & (df["lag_2"] == 0)).astype(int)
    df["spike_recency_score"] = (
        df["lag_1"] * 0.5 +
        df["lag_2"] * 0.3 +
        df["lag_3"] * 0.2
    )

    return df.dropna()