import numpy as np
import pandas as pd

def make_exog_features(df, is_train=True):
    """
    Build exogenous features for SARIMAX.
    df must have: date index, 'main_category' column, 'y' column
    """
    exog = pd.DataFrame(index=df.index)

    # ── TIME FEATURES ──
    exog["is_thursday"] = (pd.to_datetime(df.index).dayofweek == 3).astype(int)
    exog["is_q4"]       = (pd.to_datetime(df.index).quarter   == 4).astype(int)
    exog["is_weekend"]  = (pd.to_datetime(df.index).dayofweek >= 5).astype(int)
    exog["month_sin"]   = np.sin(2 * np.pi * pd.to_datetime(df.index).month / 12)
    exog["month_cos"]   = np.cos(2 * np.pi * pd.to_datetime(df.index).month / 12)

    # ── main_category FLAGS ──
    exog["is_tv_video"] = df["main_category"].str.lower().isin(
                              ["tv", "video", "tv & video"]
                          ).astype(int)

    exog["is_computer"] = df["main_category"].str.lower().isin(
                              ["computer", "computers", "computing"]
                          ).astype(int)

    exog["is_accessory"] = df["main_category"].str.lower().isin(
                               ["accessories", "accessory"]
                           ).astype(int)

    # ── main_category STATS (always computed from train_orig to avoid leakage) ──
    cat_zero_rate      = (train_orig.groupby("main_category")["y"]
                          .apply(lambda x: (x == 0).mean()))

    cat_mean_discount  = train_orig.groupby("main_category")["y"].mean()

    cat_std_discount   = train_orig.groupby("main_category")["y"].std().fillna(0)

    cat_spike_rate     = (train_orig.groupby("main_category")["is_nonzero"]
                          .mean())

    cat_median         = train_orig.groupby("main_category")["y"].median()

    exog["cat_zero_rate"]      = df["main_category"].map(cat_zero_rate).fillna(0)
    exog["cat_mean_discount"]  = df["main_category"].map(cat_mean_discount).fillna(0)
    exog["cat_std_discount"]   = df["main_category"].map(cat_std_discount).fillna(0)
    exog["cat_spike_rate"]     = df["main_category"].map(cat_spike_rate).fillna(0)
    exog["cat_median"]         = df["main_category"].map(cat_median).fillna(0)

    # ── SPIKE BOOST FEATURES ──
    # Rolling non-zero rate per main_category per month
    cat_month_spike = (train_orig
                       .assign(month=pd.to_datetime(train_orig.index).month)
                       .groupby(["main_category", "month"])["is_nonzero"]
                       .mean())

    months = pd.to_datetime(df.index).month
    exog["cat_month_spike"] = [
        cat_month_spike.get((cat, m), 0)
        for cat, m in zip(df["main_category"], months)
    ]

    # ── INTERACTION FEATURES ──
    exog["q4_tv_boost"]        = exog["is_q4"]       * exog["is_tv_video"]
    exog["thursday_spike_cat"] = exog["is_thursday"]  * exog["cat_spike_rate"]
    exog["q4_spike_cat"]       = exog["is_q4"]        * exog["cat_spike_rate"]
    exog["computer_penalty"]   = exog["is_computer"]  * exog["cat_zero_rate"]

    # ── LOG TRANSFORM main_category mean (matches log-transformed y) ──
    exog["cat_mean_log"]   = np.log1p(exog["cat_mean_discount"])
    exog["cat_median_log"] = np.log1p(exog["cat_median"])

    return exog.astype(float)