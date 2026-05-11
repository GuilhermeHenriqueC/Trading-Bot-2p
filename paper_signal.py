import os
import yfinance as yf
import pandas as pd

DATA_DIR = "Data"
NEWS_CANDIDATES_FILE = os.path.join(DATA_DIR, "news_candidates.csv")
BACKTEST_SUMMARY_FILE = os.path.join(DATA_DIR, "multi_backtest_summary.csv")
SIGNALS_OUTPUT_FILE = os.path.join(DATA_DIR, "paper_signals.csv")

INTERVAL = "5m"
PERIOD = "5d"
DROP_TRIGGER = -0.01
VOLUME_MULTIPLIER = 1.2
MIN_WIN_RATE = 55
MIN_TRADES = 3
MIN_TOTAL_RETURN = 0
MAX_SIGNALS_TO_CHECK = 15
TAKE_PROFIT = 0.02
STOP_LOSS = -0.02

os.makedirs(DATA_DIR, exist_ok=True)


def write_empty_signals(reason):
    pd.DataFrame(columns=["ticker", "signal", "reason"]).to_csv(SIGNALS_OUTPUT_FILE, index=False)
    print(reason)
    print(f"Cleared stale signals in {SIGNALS_OUTPUT_FILE}")


def load_news_candidates():
    if not os.path.exists(NEWS_CANDIDATES_FILE):
        return pd.DataFrame()

    df = pd.read_csv(NEWS_CANDIDATES_FILE)

    if "status" in df.columns:
        df = df[df["status"] == "NEWS_CANDIDATE"]

    return df


def load_backtest_summary():
    if not os.path.exists(BACKTEST_SUMMARY_FILE):
        return pd.DataFrame()

    return pd.read_csv(BACKTEST_SUMMARY_FILE)


def get_intraday_data(ticker):
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False, threads=False)

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if "Datetime" not in df.columns:
        return pd.DataFrame()

    df["date"] = df["Datetime"].dt.date
    df["time"] = df["Datetime"].dt.time

    return df


def check_live_conditions(ticker):
    df = get_intraday_data(ticker)

    if df.empty:
        return {"ticker": ticker, "signal": "SKIP", "reason": "No intraday data"}

    latest_date = df["date"].max()
    today = df[df["date"] == latest_date].copy().reset_index(drop=True)
    history = df[df["date"] != latest_date].copy()

    if today.empty or len(today) < 2:
        return {"ticker": ticker, "signal": "WAIT", "reason": "Not enough candles today"}

    latest = today.iloc[-1]
    open_price = today.iloc[0]["Open"]
    current_price = latest["Close"]
    current_volume = latest["Volume"]
    current_time = latest["Datetime"].strftime("%H:%M")
    move_from_open = (current_price - open_price) / open_price

    if not history.empty:
        avg_volume_by_time = history.groupby("time")["Volume"].mean()
        avg_volume = avg_volume_by_time.get(latest["time"], history["Volume"].mean())
    else:
        avg_volume = today["Volume"].mean()

    volume_ratio = current_volume / avg_volume if avg_volume and avg_volume > 0 else 0

    if move_from_open <= DROP_TRIGGER and volume_ratio >= VOLUME_MULTIPLIER:
        signal = "BUY"
        reason = "News candidate + promising backtest + current dip + high volume"
    elif move_from_open > DROP_TRIGGER:
        signal = "WAIT"
        reason = "No large enough dip from open yet"
    elif volume_ratio < VOLUME_MULTIPLIER:
        signal = "WAIT"
        reason = "Volume confirmation not strong enough yet"
    else:
        signal = "WAIT"
        reason = "Waiting for confirmation"

    return {
        "ticker": ticker,
        "signal": signal,
        "reason": reason,
        "time": current_time,
        "open_price": round(open_price, 4),
        "current_price": round(current_price, 4),
        "move_from_open_pct": round(move_from_open * 100, 2),
        "volume_ratio": round(volume_ratio, 2),
        "take_profit_price": round(current_price * (1 + TAKE_PROFIT), 4),
        "stop_loss_price": round(current_price * (1 + STOP_LOSS), 4),
    }


def main():
    print("Running paper_signal.py")

    news_df = load_news_candidates()
    summary_df = load_backtest_summary()

    if news_df.empty:
        write_empty_signals("No NEWS_CANDIDATE tickers found. Run news_scanner.py first.")
        return

    if summary_df.empty:
        write_empty_signals("No backtest summary found. Run multi_backtest.py first.")
        return

    candidates = news_df[["ticker"]].drop_duplicates().merge(summary_df, on="ticker", how="inner")

    if candidates.empty:
        write_empty_signals(f"No matching tickers between {NEWS_CANDIDATES_FILE} and {BACKTEST_SUMMARY_FILE}.")
        return

    candidates = candidates[
        (candidates["trades"] >= MIN_TRADES)
        & (candidates["win_rate"] >= MIN_WIN_RATE)
        & (candidates["total_return"] > MIN_TOTAL_RETURN)
        & (candidates["status"] == "PROMISING")
    ]

    if candidates.empty:
        write_empty_signals("No candidates passed paper signal requirements.")
        return

    candidates = candidates.sort_values(
        ["win_rate", "total_return", "average_return", "trades"],
        ascending=[False, False, False, False],
    ).head(MAX_SIGNALS_TO_CHECK)

    print(f"Checking top {len(candidates)} live candidates out of {MAX_SIGNALS_TO_CHECK} max.")

    signals = []

    for ticker in candidates["ticker"]:
        print(f"Checking live signal for {ticker}...")
        signals.append(check_live_conditions(ticker))

    signals_df = pd.DataFrame(signals)
    signals_df.to_csv(SIGNALS_OUTPUT_FILE, index=False)

    print("\nPaper Signals")
    print(signals_df.to_string(index=False))
    print(f"\nSaved signals to {SIGNALS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
