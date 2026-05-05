import os
import pandas as pd
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, AssetClass, AssetStatus

load_dotenv()

DATA_DIR = "Data"
SIGNALS_FILE = os.path.join(DATA_DIR, "paper_signals.csv")
USE_ACCOUNT_PERCENTAGE = True
ACCOUNT_PERCENTAGE_PER_TRADE = 0.95
MAX_DOLLARS_PER_TRADE = 1000
MIN_CASH_BUFFER = 25
ALLOW_FRACTIONAL = False

TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.02"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))
PRICE_BUFFER = 0.01

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


def get_latest_asset_price(client, ticker, fallback_price):
    try:
        asset = client.get_asset(ticker)
    except Exception as error:
        print(f"Could not load asset data for {ticker}. Using signal price. Error: {error}")
        return fallback_price

    if not asset.tradable:
        print(f"Skipping {ticker}: asset is not tradable on Alpaca.")
        return None

    try:
        if hasattr(asset, "price_increment") and asset.price_increment:
            return round(float(fallback_price), 2)
    except Exception:
        pass

    return round(float(fallback_price), 2)


def calculate_bracket_prices(entry_price):
    take_profit_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)
    stop_loss_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)

    minimum_take_profit = round(entry_price + PRICE_BUFFER, 2)
    maximum_stop_loss = round(entry_price - PRICE_BUFFER, 2)

    if take_profit_price < minimum_take_profit:
        take_profit_price = minimum_take_profit

    if stop_loss_price >= entry_price:
        stop_loss_price = maximum_stop_loss

    return take_profit_price, stop_loss_price


def submit_bracket_order(client, row):
    ticker = str(row["ticker"])
    signal_price = float(row["current_price"])
    current_price = get_latest_asset_price(client, ticker, signal_price)

    if current_price is None:
        return

    take_profit_price, stop_loss_price = calculate_bracket_prices(current_price)

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
        print("Signal price:", signal_price)
        print("Order price used:", current_price)
        print("Take profit used:", take_profit_price)
        print("Stop loss used:", stop_loss_price)
        return

    print(f"Submitted paper bracket order for {ticker}")
    print("Order ID:", order.id)
    print("Entry type: market buy")
    print("Quantity:", qty)
    print("Approx allocation:", round(qty * current_price, 2))
    print("Signal price:", signal_price)
    print("Order price used:", current_price)
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
