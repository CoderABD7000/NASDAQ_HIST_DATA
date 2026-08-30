import glob
import os
import zipfile
import numpy as np
import pandas as pd

UNZIP_DIR = "./unzipped"
COL_NAMES = ["date", "time", "open", "high", "low", "close"]
os.makedirs(UNZIP_DIR, exist_ok=True)

# 1. Extract raw archives
zip_files = glob.glob("*.zip")
print(f"Extracting {len(zip_files)} raw archive files...")
for zf in zip_files:
    with zipfile.ZipFile(zf, "r") as zip_ref:
        zip_ref.extractall(UNZIP_DIR)

# 2. Parse and merge into continuous 1m timeline
all_files = glob.glob(os.path.join(UNZIP_DIR, "**/*.csv"), recursive=True)
if not all_files:
    all_files = glob.glob(os.path.join(UNZIP_DIR, "**/*.txt"), recursive=True)

df_list = []
for file_path in sorted(all_files):
    with open(file_path, "r") as f:
        first_line = f.readline()
        sep = ";" if ";" in first_line else ","
    try:
        temp_df = pd.read_csv(file_path, sep=sep, header=None)
        if isinstance(temp_df.iloc[0, 2], str):
            temp_df = pd.read_csv(file_path, sep=sep, header=0)
        temp_df = temp_df.iloc[:, :6]
        temp_df.columns = COL_NAMES
        df_list.append(temp_df)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")

master_1m = pd.concat(df_list, ignore_index=True)
master_1m["datetime_str"] = (
    master_1m["date"].astype(str) + " " + master_1m["time"].astype(str)
)
master_1m["datetime"] = pd.to_datetime(
    master_1m["datetime_str"], errors="coerce"
)
master_1m.dropna(subset=["datetime"], inplace=True)
master_1m.drop_duplicates(subset=["datetime"], inplace=True)
master_1m.sort_values(by="datetime", inplace=True)
master_1m.set_index("datetime", inplace=True)

# 3. Resample base 5m bars
df_5m = (
    master_1m.resample("5min")
    .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    .dropna()
)


# 4. Compute ADX, DI+, DI- (14-period Wilder smoothing)
def compute_adx_system(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean()
    pos_di = (
        100
        * pd.Series(pos_dm, index=df.index)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
        / atr
    )
    neg_di = (
        100
        * pd.Series(neg_dm, index=df.index)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
        / atr
    )

    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    df["adx_14"] = adx.round(2)
    df["di_plus_14"] = pos_di.round(2)
    df["di_minus_14"] = neg_di.round(2)
    return df


df_5m = compute_adx_system(df_5m)

# 5. Compute Fast Frozen POC Levels
df_5m["date_group"] = df_5m.index.date
df_5m["h4_group"] = df_5m.index.floor("4h")
df_5m["week_group"] = df_5m.index.to_period("W").dt.start_time


def get_first_mode(series):
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan


daily_poc_map = df_5m.groupby("date_group")["close"].agg(get_first_mode)
h4_poc_map = df_5m.groupby("h4_group")["close"].agg(get_first_mode)
weekly_poc_map = df_5m.groupby("week_group")["close"].agg(get_first_mode)

df_5m["daily_poc"] = df_5m["date_group"].map(daily_poc_map.shift(1))
df_5m["h4_poc"] = df_5m["h4_group"].map(h4_poc_map.shift(1))
df_5m["weekly_poc"] = df_5m["week_group"].map(weekly_poc_map.shift(1))

# 6. Format and save
df_5m["date"] = df_5m.index.strftime("%Y.%m.%d")
df_5m["time"] = df_5m.index.strftime("%H:%M")

final_cols = [
    "date",
    "time",
    "open",
    "high",
    "low",
    "close",
    "daily_poc",
    "h4_poc",
    "weekly_poc",
    "adx_14",
    "di_plus_14",
    "di_minus_14",
]

df_out = df_5m[final_cols].dropna()
output_file = "NAS100_2019_2026_5m_Enriched_Metrics.csv"
df_out.to_csv(output_file, index=False)
print(f"Successfully generated {output_file} with {len(df_out):,} rows!")
