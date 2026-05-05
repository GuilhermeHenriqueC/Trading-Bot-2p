import os
import yfinance as yf
import pandas as pd

DATA_DIR = "Data"
TICKER = "TMC"
DROP_TRIGGER = -0.01
VOLUME_MULTIPLIER = 1.2
TAKE_PROFIT = 0.02
STOP_LOSS = -0.02
USE_TIME_FILTER = True
MIN_TIME_TRADES = 2
MIN_TIME_WIN_RATE = 55

os.makedirs(DATA_DIR, exist_ok=True)

analysis_file = os.path.join(DATA_DIR, f"{TICKER}_time_analysis.csv")
results_file = os.path.join(DATA_DIR, f"{TICKER}_backtest_results.csv")
allowed_entry_times = None

if USE_TIME_FILTER and os.path.exists(analysis_file):
    time_analysis = pd.read_csv(analysis_file)
    strong_times = time_analysis[
        (time_analysis["trades"] >= MIN_TIME_TRADES)
        & (time_analysis["win_rate"] >= MIN_TIME_WIN_RATE)
    ]
    allowed_entry_times = set(strong_times["entry_clock"].astype(str))

    if allowed_entry_times:
        print("Using time filter:", sorted(allowed_entry_times))
    else:
        print("No strong entry times found in analysis file. Time filter disabled.")
        allowed_entry_times = None
elif USE_TIME_FILTER:
    print(f"No {analysis_file} found yet. Time filter disabled.")

df = yf.download(TICKER, period="60d", interval="5m", auto_adjust=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

df["date"] = df["Datetime"].dt.date
df["time"] = df["Datetime"].dt.time

df["open_day"] = df.groupby("date")["Open"].transform("first")
df["move_from_open"] = (df["Close"] - df["open_day"]) / df["open_day"] * 100

avg_volume_by_time = df.groupby("time")["Volume"].mean()
df["avg_volume_time"] = df["time"].map(avg_volume_by_time)
df["volume_ratio"] = df["Volume"] / df["avg_volume_time"]

pattern = df.groupby("time")["move_from_open"].mean()

print(pattern.sort_values().head(10))
print(pattern.sort_values().tail(10))


trades = []

for date, day in df.groupby("date"):
    day = day.reset_index(drop=True)

    for i in range(1, len(day)):
        entry_clock = day.loc[i, "Datetime"].strftime("%H:%M")

        if allowed_entry_times is not None and entry_clock not in allowed_entry_times:
            continue

        drop_from_open = (day.loc[i, "Close"] - day.loc[0, "Open"]) / day.loc[0, "Open"]
        volume_ratio = day.loc[i, "volume_ratio"]

        if drop_from_open <= DROP_TRIGGER and volume_ratio >= VOLUME_MULTIPLIER:
            entry = day.loc[i, "Close"]
            entry_time = day.loc[i, "Datetime"]
            exit_price = None
            exit_time = None

            for j in range(i + 1, len(day)):
                current = day.loc[j, "Close"]
                ret = (current - entry) / entry

                if ret >= TAKE_PROFIT:
                    exit_price = current
                    exit_time = day.loc[j, "Datetime"]
                    trades.append([date, entry_time, exit_time, entry, exit_price, "WIN", ret, volume_ratio])
                    break

                if ret <= STOP_LOSS:
                    exit_price = current
                    exit_time = day.loc[j, "Datetime"]
                    trades.append([date, entry_time, exit_time, entry, exit_price, "LOSS", ret, volume_ratio])
                    break

            break

results = pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "entry_price", "exit_price", "result", "return", "volume_ratio"])

if results.empty:
    print("No trades found with the current strategy rules.")
else:
    results.to_csv(results_file, index=False)

    wins = (results["result"] == "WIN").sum()
    losses = (results["result"] == "LOSS").sum()
    win_rate = wins / len(results)
    total_return = results["return"].sum()
    average_return = results["return"].mean()

    print(results)
    print("\nSummary")
    print("Ticker:", TICKER)
    print("Trades:", len(results))
    print("Wins:", wins)
    print("Losses:", losses)
    print("Win rate:", round(win_rate * 100, 2), "%")
    print("Total return:", round(total_return * 100, 2), "%")
    print("Average return per trade:", round(average_return * 100, 2), "%")
    print(f"Saved results to {results_file}")
