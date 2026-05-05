import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import pandas as pd

DATA_DIR = "Data"
SIGNALS_FILE = f"{DATA_DIR}/paper_signals.csv"
SUMMARY_FILE = f"{DATA_DIR}/multi_backtest_summary.csv"
NEWS_FILE = f"{DATA_DIR}/news_candidates.csv"

PRE_TRADE_STEPS = [
    ("News scanner", "news_scanner.py"),
    ("Multi backtest", "multi_backtest.py"),
]

SIGNAL_STEP = ("Paper signal", "paper_signal.py")
TRADE_STEP = ("Paper trade", "paper_trade.py")

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)
DEFAULT_CHECK_SECONDS = 60


def run_step(name, script):
    print(f"\n===== Running {name} =====")
    result = subprocess.run([sys.executable, script], text=True)

    if result.returncode != 0:
        print(f"\nStopped: {script} failed.")
        sys.exit(result.returncode)


def print_csv_preview(title, file_path):
    print(f"\n===== {title} =====")

    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    if df.empty:
        print("No rows found.")
        return df

    print(df.to_string(index=False))
    return df


def get_buy_signals(signals):
    if signals.empty or "signal" not in signals.columns:
        return pd.DataFrame()

    return signals[signals["signal"] == "BUY"]


def print_proposed_trades(buy_signals):
    print("\n===== Proposed Trades =====")
    columns = [
        "ticker",
        "signal",
        "current_price",
        "take_profit_price",
        "stop_loss_price",
        "move_from_open_pct",
        "volume_ratio",
        "reason",
    ]
    available_columns = [column for column in columns if column in buy_signals.columns]
    print(buy_signals[available_columns].to_string(index=False))


def market_is_open():
    now = datetime.now(MARKET_TZ)

    if now.weekday() >= 5:
        return False

    current_time = now.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


def wait_until_market_open(check_seconds):
    while not market_is_open():
        now = datetime.now(MARKET_TZ)
        print(f"Market is closed. Current New York time: {now:%Y-%m-%d %H:%M:%S}. Checking again in {check_seconds} seconds.")
        time.sleep(check_seconds)


def run_pre_trade_steps():
    os.makedirs(DATA_DIR, exist_ok=True)

    for name, script in PRE_TRADE_STEPS:
        run_step(name, script)

    print_csv_preview("News Candidates", NEWS_FILE)
    print_csv_preview("Backtest Summary", SUMMARY_FILE)


def run_signal_check():
    run_step(*SIGNAL_STEP)
    signals = print_csv_preview("Paper Signals", SIGNALS_FILE)
    return get_buy_signals(signals)


def run_manual_mode():
    run_pre_trade_steps()
    buy_signals = run_signal_check()

    if buy_signals.empty:
        print("\nNo BUY signals. Trade step skipped.")
        return

    print_proposed_trades(buy_signals)
    answer = input("\nSubmit these paper trades to Alpaca? Type YES to confirm: ").strip()

    if answer != "YES":
        print("Trade step cancelled.")
        return

    run_step(*TRADE_STEP)
    print("\nBot pipeline finished.")


def run_auto_mode(check_seconds):
    run_pre_trade_steps()
    wait_until_market_open(check_seconds)
    traded_tickers = set()

    while market_is_open():
        buy_signals = run_signal_check()

        if not buy_signals.empty:
            if "ticker" in buy_signals.columns:
                new_buy_signals = buy_signals[~buy_signals["ticker"].isin(traded_tickers)]
            else:
                new_buy_signals = buy_signals

            if not new_buy_signals.empty:
                print_proposed_trades(new_buy_signals)
                print("\nAUTO mode enabled. Submitting paper trades to Alpaca.")
                run_step(*TRADE_STEP)

                if "ticker" in new_buy_signals.columns:
                    traded_tickers.update(new_buy_signals["ticker"].dropna().astype(str).tolist())

                print("\nTrade step finished. Bot will keep checking until market close.")
            else:
                print("\nBUY signals found, but all matching tickers were already traded in this session.")

        now = datetime.now(MARKET_TZ)
        print(f"\nCurrent New York time: {now:%Y-%m-%d %H:%M:%S}. Checking again in {check_seconds} seconds.")
        time.sleep(check_seconds)

    print("\nMarket closed. Auto mode stopped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--check-seconds", type=int, default=DEFAULT_CHECK_SECONDS)
    args = parser.parse_args()

    if args.auto:
        run_auto_mode(args.check_seconds)
    else:
        run_manual_mode()


if __name__ == "__main__":
    main()
