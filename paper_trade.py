import os
import pandas as pd
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

load_dotenv()

DATA_DIR = "Data"
SIGNALS_FILE = os.path.join(DATA_DIR, "paper_signals.csv")
USE_ACCOUNT_PERCENTAGE = True
ACCOUNT_PERCENTAGE_PER_TRADE = 0.95
MAX_DOLLARS_PER_TRADE = 1000
MIN_CASH_BUFFER = 25
ALLOW_FRACTIONAL = False

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


def get_client():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("Missing Alpaca keys in .env")
        print("Add ALPACA_API_KEY and ALPACA_SECRET_KEY to your .env file.")
        raise SystemExit

    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)


def load_buy_signals():
    if not os.path.exists(SIGNALS_FILE):
        print(f"Missing {SIGNALS_FILE}. Run paper_signal.py first.")
        return pd.DataFrame()

    signals = pd.read_csv(SIGNALS_FILE)

    if "signal" not in signals.columns:
        print(f"{SIGNALS_FILE} has no signal column.")
        return pd.DataFrame()

    return signals[signals["signal"] == "BUY"].copy()


def already_has_position(client, ticker):
    try:
        client.get_open_position(ticker)
        return True
    except Exception:
        return False


def calculate_trade_dollars(client):
    account = client.get_account()
    buying_power = float(account.buying_power)

    if USE_ACCOUNT_PERCENTAGE:
        trade_dollars = buying_power * ACCOUNT_PERCENTAGE_PER_TRADE
    else:
        trade_dollars = MAX_DOLLARS_PER_TRADE

    trade_dollars = min(trade_dollars, buying_power - MIN_CASH_BUFFER)

    if trade_dollars <= 0:
        return 0

    return round(trade_dollars, 2)


def submit_bracket_order(client, row):
    ticker = row["ticker"]
    current_price = float(row["current_price"])
    take_profit_price = round(float(row["take_profit_price"]), 2)
    stop_loss_price = round(float(row["stop_loss_price"]), 2)

    if already_has_position(client, ticker):
        print(f"Skipping {ticker}: already has open position.")
        return

    if ALLOW_FRACTIONAL:
        print(f"Skipping {ticker}: Alpaca does not allow fractional/notional bracket orders.")
        return

    trade_dollars = calculate_trade_dollars(client)

    if trade_dollars <= 0:
        print(f"Skipping {ticker}: not enough buying power after cash buffer.")
        return

    qty = int(trade_dollars // current_price)

    if qty <= 0:
        print(f"Skipping {ticker}: price too high for ${trade_dollars} allocation.")
        return

    order_data = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
        stop_loss=StopLossRequest(stop_price=stop_loss_price),
    )

    try:
        order = client.submit_order(order_data)
    except Exception as error:
        print(f"Failed to submit order for {ticker}:", error)
        return

    print(f"Submitted paper bracket order for {ticker}")
    print("Order ID:", order.id)
    print("Entry type: market buy")
    print("Quantity:", qty)
    print("Approx allocation:", round(qty * current_price, 2))
    print("Take profit:", take_profit_price)
    print("Stop loss:", stop_loss_price)


def main():
    buy_signals = load_buy_signals()

    if buy_signals.empty:
        print("No BUY signals found. Nothing to trade.")
        return

    client = get_client()
    account = client.get_account()

    print("Alpaca paper account status:", account.status)
    print("Buying power:", account.buying_power)
    print("Use account percentage:", USE_ACCOUNT_PERCENTAGE)
    print("Account percentage per trade:", ACCOUNT_PERCENTAGE_PER_TRADE)
    print("Fixed max dollars per trade:", MAX_DOLLARS_PER_TRADE)
    print("Minimum cash buffer:", MIN_CASH_BUFFER)

    for _, row in buy_signals.iterrows():
        submit_bracket_order(client, row)


if __name__ == "__main__":
    main()
