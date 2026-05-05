import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

DATA_DIR = "Data"
SIGNALS_FILE = f"{DATA_DIR}/paper_signals.csv"
SUMMARY_FILE = f"{DATA_DIR}/multi_backtest_summary.csv"
NEWS_FILE = f"{DATA_DIR}/news_candidates.csv"
TRADED_FILE = f"{DATA_DIR}/traded_today.json"
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "5"))

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

load_dotenv()


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


# ==== NEW: Trade state and filtering logic ====
def today_market_date():
    return datetime.now(MARKET_TZ).strftime("%Y-%m-%d")


def load_traded_state():
    if not os.path.exists(TRADED_FILE):
        return {"date": today_market_date(), "tickers": []}

    try:
        with open(TRADED_FILE, "r", encoding="utf-8") as file:
            state = json.load(file)
    except (json.JSONDecodeError, OSError):
        return {"date": today_market_date(), "tickers": []}

    if state.get("date") != today_market_date():
        return {"date": today_market_date(), "tickers": []}

    return state


def save_traded_state(state):
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(TRADED_FILE, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


def get_open_position_tickers():
    try:
        import alpaca_trade_api as tradeapi
    except ImportError:
        print("\nAlpaca package not installed. Position check skipped.")
        return set()

    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL") or os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"

    if not api_key or not secret_key:
        print("\nAlpaca API keys missing. Position check skipped.")
        return set()

    try:
        api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")
        positions = api.list_positions()
    except Exception as error:
        print(f"\nCould not load Alpaca open positions. Position check skipped. Error: {error}")
        return set()

    return {position.symbol for position in positions}


def filter_new_buy_signals(buy_signals, traded_tickers, held_tickers):
    if buy_signals.empty or "ticker" not in buy_signals.columns:
        return buy_signals

    tickers = buy_signals["ticker"].astype(str)
    return buy_signals[~tickers.isin(traded_tickers) & ~tickers.isin(held_tickers)]


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

    traded_state = load_traded_state()
    traded_tickers = set(traded_state.get("tickers", []))

    print(f"\nDaily max trades: {MAX_TRADES_PER_DAY}")
    print(f"Already traded today: {sorted(traded_tickers) if traded_tickers else 'none'}")

    while market_is_open():
        traded_state = load_traded_state()
        traded_tickers = set(traded_state.get("tickers", []))

        if len(traded_tickers) >= MAX_TRADES_PER_DAY:
            print(f"\nMax trades reached for today: {MAX_TRADES_PER_DAY}. Checking again until market close, but no new trades will be submitted.")
            time.sleep(check_seconds)
            continue

        buy_signals = run_signal_check()

        if not buy_signals.empty:
            held_tickers = get_open_position_tickers()
            new_buy_signals = filter_new_buy_signals(buy_signals, traded_tickers, held_tickers)
            remaining_trade_slots = MAX_TRADES_PER_DAY - len(traded_tickers)
            new_buy_signals = new_buy_signals.head(remaining_trade_slots)

            if not new_buy_signals.empty:
                print_proposed_trades(new_buy_signals)
                print("\nAUTO mode enabled. Submitting paper trades to Alpaca.")
                run_step(*TRADE_STEP)

                if "ticker" in new_buy_signals.columns:
                    traded_tickers.update(new_buy_signals["ticker"].dropna().astype(str).tolist())
                    save_traded_state({"date": today_market_date(), "tickers": sorted(traded_tickers)})

                print("\nTrade step finished. Bot will keep checking until market close.")
            else:
                print("\nBUY signals found, but they were already traded today, already held in Alpaca, or max trade slots are full.")

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
