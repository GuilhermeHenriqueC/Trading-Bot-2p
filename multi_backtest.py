import os
import yfinance as yf
import pandas as pd

DATA_DIR = "Data"
DEFAULT_TICKERS = [
    "AAPL",
    "TSLA",
    "NVDA",
    "AMD",
    "PLTR",
    "SOFI",
    "TMC",
    "RIVN",
    "COIN",
    "MARA",
]

NEWS_CANDIDATES_FILE = os.path.join(DATA_DIR, "news_candidates.csv")
BACKTEST_SUMMARY_FILE = os.path.join(DATA_DIR, "multi_backtest_summary.csv")
BACKTEST_TRADES_FILE = os.path.join(DATA_DIR, "multi_backtest_trades.csv")
USE_NEWS_CANDIDATES = True

PERIOD = "60d"
INTERVAL = "5m"
DROP_TRIGGER = -0.01
VOLUME_MULTIPLIER = 1.2
TAKE_PROFIT = 0.02
STOP_LOSS = -0.02
MIN_TRADES = 3
MIN_WIN_RATE = 55
USE_TIME_FILTER = True
MIN_TIME_TRADES = 2
MIN_TIME_WIN_RATE = 55
FALLBACK_TO_DEFAULT_TICKERS = False

os.makedirs(DATA_DIR, exist_ok=True)


def load_tickers():
    if USE_NEWS_CANDIDATES and os.path.exists(NEWS_CANDIDATES_FILE):
        news_df = pd.read_csv(NEWS_CANDIDATES_FILE)

        if "ticker" not in news_df.columns:
            print(f"{NEWS_CANDIDATES_FILE} found, but it has no ticker column. Using default tickers.")
            return DEFAULT_TICKERS

        if "status" in news_df.columns:
            news_df = news_df[news_df["status"] == "NEWS_CANDIDATE"]

        tickers = sorted(news_df["ticker"].dropna().astype(str).str.upper().unique())

        if tickers:
            print("Using news candidate tickers:", tickers)
            return tickers

        if FALLBACK_TO_DEFAULT_TICKERS:
            print(f"{NEWS_CANDIDATES_FILE} found, but no NEWS_CANDIDATE tickers. Using default tickers.")
            return DEFAULT_TICKERS

        print(f"{NEWS_CANDIDATES_FILE} found, but no NEWS_CANDIDATE tickers. No tickers to test.")
        return []

    if FALLBACK_TO_DEFAULT_TICKERS:
        print(f"No {NEWS_CANDIDATES_FILE} found. Using default tickers.")
        return DEFAULT_TICKERS

    print(f"No {NEWS_CANDIDATES_FILE} found. Run news_scanner.py first.")
    return []


TICKERS = load_tickers()

if not TICKERS:
    pd.DataFrame(columns=[
        "ticker",
        "trades",
        "wins",
        "losses",
        "win_rate",
        "total_return",
        "average_return",
        "status",
        "source",
        "time_filter",
        "allowed_times",
    ]).to_csv(BACKTEST_SUMMARY_FILE, index=False)
    print(f"No tickers to backtest. Saved empty {BACKTEST_SUMMARY_FILE}")
    raise SystemExit

all_summaries = []
all_trades = []


def download_data(ticker):
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=False, progress=False)

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if "Datetime" not in df.columns:
        return pd.DataFrame()

    df["date"] = df["Datetime"].dt.date
    df["time"] = df["Datetime"].dt.time
    df["open_day"] = df.groupby("date")["Open"].transform("first")
    df["move_from_open"] = (df["Close"] - df["open_day"]) / df["open_day"] * 100

    avg_volume_by_time = df.groupby("time")["Volume"].mean()
    df["avg_volume_time"] = df["time"].map(avg_volume_by_time)
    df["volume_ratio"] = df["Volume"] / df["avg_volume_time"]

    return df


def backtest_ticker(ticker, df, allowed_entry_times=None):
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

                for j in range(i + 1, len(day)):
                    current = day.loc[j, "Close"]
                    ret = (current - entry) / entry

                    if ret >= TAKE_PROFIT:
                        trades.append([
                            ticker,
                            date,
                            entry_time,
                            day.loc[j, "Datetime"],
                            entry_clock,
                            entry,
                            current,
                            "WIN",
                            ret,
                            volume_ratio,
                        ])
                        break

                    if ret <= STOP_LOSS:
                        trades.append([
                            ticker,
                            date,
                            entry_time,
                            day.loc[j, "Datetime"],
                            entry_clock,
                            entry,
                            current,
                            "LOSS",
                            ret,
                            volume_ratio,
                        ])
                        break

                break

    columns = [
        "ticker",
        "date",
        "entry_time",
        "exit_time",
        "entry_clock",
        "entry_price",
        "exit_price",
        "result",
        "return",
        "volume_ratio",
    ]

    return pd.DataFrame(trades, columns=columns)


def find_allowed_entry_times(trades):
    if trades.empty:
        return None

    time_analysis = (
        trades.groupby("entry_clock")
        .agg(
            trades=("result", "count"),
            wins=("result", lambda x: (x == "WIN").sum()),
            win_rate=("result", lambda x: (x == "WIN").mean() * 100),
            total_return=("return", lambda x: x.sum() * 100),
            average_return=("return", lambda x: x.mean() * 100),
        )
        .reset_index()
    )

    strong_times = time_analysis[
        (time_analysis["trades"] >= MIN_TIME_TRADES)
        & (time_analysis["win_rate"] >= MIN_TIME_WIN_RATE)
        & (time_analysis["total_return"] > 0)
    ]

    if strong_times.empty:
        return None

    return set(strong_times["entry_clock"].astype(str))


def summarize_ticker(ticker, trades):
    if trades.empty:
        return {
            "ticker": ticker,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_return": 0,
            "average_return": 0,
            "status": "NO_TRADES",
        }

    wins = (trades["result"] == "WIN").sum()
    losses = (trades["result"] == "LOSS").sum()
    win_rate = wins / len(trades) * 100
    total_return = trades["return"].sum() * 100
    average_return = trades["return"].mean() * 100

    if len(trades) < MIN_TRADES:
        status = "LOW_SAMPLE"
    elif win_rate >= MIN_WIN_RATE and total_return > 0:
        status = "PROMISING"
    else:
        status = "WEAK"

    return {
        "ticker": ticker,
        "trades": len(trades),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(win_rate, 2),
        "total_return": round(total_return, 2),
        "average_return": round(average_return, 2),
        "status": status,
    }


for ticker in TICKERS:
    print(f"Testing {ticker}...")
    df = download_data(ticker)

    if df.empty:
        print(f"No data for {ticker}")
        summary_row = summarize_ticker(ticker, pd.DataFrame())
        summary_row["source"] = "NEWS" if USE_NEWS_CANDIDATES and os.path.exists(NEWS_CANDIDATES_FILE) else "DEFAULT"
        all_summaries.append(summary_row)
        continue

    raw_trades = backtest_ticker(ticker, df)
    allowed_entry_times = find_allowed_entry_times(raw_trades) if USE_TIME_FILTER else None

    if allowed_entry_times:
        print(f"Using time filter for {ticker}: {sorted(allowed_entry_times)}")
        trades = backtest_ticker(ticker, df, allowed_entry_times)
    else:
        print(f"No strong time filter for {ticker}. Using raw strategy.")
        trades = raw_trades

    summary_row = summarize_ticker(ticker, trades)
    summary_row["source"] = "NEWS" if USE_NEWS_CANDIDATES and os.path.exists(NEWS_CANDIDATES_FILE) else "DEFAULT"
    summary_row["time_filter"] = "YES" if allowed_entry_times else "NO"
    summary_row["allowed_times"] = ", ".join(sorted(allowed_entry_times)) if allowed_entry_times else ""
    all_summaries.append(summary_row)

    if not trades.empty:
        all_trades.append(trades)

summary = pd.DataFrame(all_summaries)
summary = summary.sort_values(
    ["status", "source", "time_filter", "win_rate", "total_return", "trades"],
    ascending=[True, True, True, False, False, False],
)

summary.to_csv(BACKTEST_SUMMARY_FILE, index=False)

if all_trades:
    trades_output = pd.concat(all_trades, ignore_index=True)
    trades_output.to_csv(BACKTEST_TRADES_FILE, index=False)

print("\nMulti Backtest Summary")
print(summary.to_string(index=False))
print("\nSettings")
print("MIN_TRADES:", MIN_TRADES)
print("MIN_WIN_RATE:", MIN_WIN_RATE)
print("FALLBACK_TO_DEFAULT_TICKERS:", FALLBACK_TO_DEFAULT_TICKERS)
print(f"\nSaved summary to {BACKTEST_SUMMARY_FILE}")

if all_trades:
    print(f"Saved all trades to {BACKTEST_TRADES_FILE}")
else:
    print("No trades were found for any ticker.")
