import os
import re
import requests
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DATA_DIR = "Data"
CATEGORY = "general"
DAYS_BACK = 7
SYMBOL_CACHE_FILE = os.path.join(DATA_DIR, "finnhub_us_symbols.csv")
NEWS_DEBUG_FILE = os.path.join(DATA_DIR, "news_debug.csv")
NEWS_CANDIDATES_FILE = os.path.join(DATA_DIR, "news_candidates.csv")
NEWS_ARTICLES_FILE = os.path.join(DATA_DIR, "news_articles.csv")
MIN_COMPANY_NAME_LENGTH = 5
MIN_NEWS_SCORE = 1
MIN_ARTICLES = 1
MIN_AVG_VOLUME = 500_000
MIN_LAST_PRICE = 1
EXCLUDE_OTC_STYLE_TICKERS = True
MIN_TICKER_CONFIDENCE = "HIGH"
MIN_NAME_WORD_LENGTH = 4
USE_GEMINI_SENTIMENT = True
MIN_GEMINI_CONFIDENCE = 70
GEMINI_MODEL_CACHE = None

GEMINI_FALLBACK_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]

ENABLE_MOVER_CANDIDATES = True
TOP_MOVER_LIMIT = 10
MIN_YESTERDAY_RETURN = 0.02
MIN_WEEK_RETURN = 0.04

YAHOO_SCREENER_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
YAHOO_GAINERS_COUNT = 100
YAHOO_GAINERS_SCR_IDS = [
    "day_gainers",
    "small_cap_gainers",
    "most_actives",
]

os.makedirs(DATA_DIR, exist_ok=True)

def get_available_gemini_models():
    global GEMINI_MODEL_CACHE

    if GEMINI_MODEL_CACHE is not None:
        return GEMINI_MODEL_CACHE

    if not GEMINI_API_KEY:
        GEMINI_MODEL_CACHE = []
        return GEMINI_MODEL_CACHE

    url = "https://generativelanguage.googleapis.com/v1beta/models"
    params = {"key": GEMINI_API_KEY}

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        models = []

        for model in data.get("models", []):
            model_name = model.get("name", "").replace("models/", "")
            methods = model.get("supportedGenerationMethods", [])

            if "generateContent" in methods and "embedding" not in model_name.lower():
                models.append(model_name)

        preferred = [model for model in GEMINI_FALLBACK_MODELS if model in models]
        others = [model for model in models if model not in preferred]
        GEMINI_MODEL_CACHE = preferred + others
        return GEMINI_MODEL_CACHE

    except Exception as error:
        print("Could not fetch Gemini model list:", type(error).__name__)
        GEMINI_MODEL_CACHE = GEMINI_FALLBACK_MODELS
        return GEMINI_MODEL_CACHE

COMMON_FALSE_TICKERS = {
    "A", "AI", "API", "CEO", "CFO", "COO", "USA", "US", "UK", "EU", "SEC", "FDA", "IPO", "ETF", "EPS",
    "GDP", "CPI", "FED", "FOMC", "NYSE", "NASDAQ", "SPY", "QQQ", "USD", "PE", "PR", "IR", "IT", "EV",
    "THE", "AND", "FOR", "ARE", "YOU", "NEW", "OLD", "CAN", "MAY", "TOP", "LOW", "HIGH", "BIG",
    "BE", "BY", "HAS", "HAD", "WAS", "WAR", "DROP", "TECH", "BEST", "REAL", "LIVE", "NEWS", "DATA",
    "ONE", "TWO", "BUY", "SELL", "CALL", "PUT", "OPEN", "CLOSE", "GREEN", "RED", "CEO", "CIO",
}

POSITIVE_WORDS = [
    "beats", "beat", "raises", "raised", "upgrade", "upgraded", "surges", "jumps", "rises", "gain", "gains",
    "record", "growth", "profit", "profits", "revenue growth", "strong demand", "contract", "partnership",
    "approval", "launch", "expands", "buy rating", "outperform", "bullish", "positive", "higher", "tops estimates",
]

NEGATIVE_WORDS = [
    "misses", "miss", "cuts", "cut", "downgrade", "downgraded", "falls", "drops", "plunges", "loss", "losses",
    "lawsuit", "investigation", "recall", "bankruptcy", "weak demand", "layoffs", "sell rating", "underperform",
    "bearish", "negative", "lower", "warns", "warning",
]

HIGH_IMPACT_WORDS = [
    "earnings", "guidance", "merger", "acquisition", "contract", "partnership", "analyst",
    "upgrade", "downgrade", "revenue", "profit", "forecast", "approval", "clinical trial", "buyout",
]


def classify_with_gemini(ticker, headline, summary):
    if not USE_GEMINI_SENTIMENT:
        return "UNKNOWN", 0, "Gemini sentiment disabled"

    if not GEMINI_API_KEY:
        return "UNKNOWN", 0, "Missing GEMINI_API_KEY"

    params = {"key": GEMINI_API_KEY}

    prompt = f"""
You are analyzing stock trading news for a short-term intraday trading bot.

Ticker: {ticker}
Headline: {headline}
Summary: {summary}

Classify the likely short-term impact of this news specifically for the ticker, not for another company mentioned in the same headline.
If the article says the stock sinks, falls, drops, is pressured, loses, or is hurt by competition, classify it as NEGATIVE.
If the article is about another company and only indirectly mentions this ticker, classify it as NEUTRAL unless the impact is clearly positive or negative for this ticker.

Return only valid JSON with exactly these keys:
{{
  "sentiment": "POSITIVE" or "NEGATIVE" or "NEUTRAL",
  "confidence": 0-100,
  "reason": "short reason"
}}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 120,
        },
    }

    last_error = None
    available_models = get_available_gemini_models()

    if not available_models:
        return "UNKNOWN", 0, "No available Gemini generateContent models for this API key"

    for model in available_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        try:
            response = requests.post(url, params=params, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            text = text.replace("```json", "").replace("```", "").strip()

            import json
            result = json.loads(text)

            sentiment = str(result.get("sentiment", "NEUTRAL")).upper()
            confidence = int(result.get("confidence", 0))
            reason = str(result.get("reason", ""))

            if sentiment not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
                sentiment = "NEUTRAL"

            return sentiment, confidence, reason

        except Exception as error:
            last_error = f"{type(error).__name__} using model {model}"
            continue

    return "UNKNOWN", 0, f"Gemini error: {last_error}"


def score_text(text):
    text = text.lower()
    score = 0

    for word in POSITIVE_WORDS:
        if word in text:
            score += 1

    for word in NEGATIVE_WORDS:
        if word in text:
            score -= 1

    for word in HIGH_IMPACT_WORDS:
        if word in text:
            score += 1

    return score


def parse_finnhub_datetime(timestamp):
    if not timestamp:
        return None

    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
    except (ValueError, TypeError):
        return None


def is_recent_article(timestamp):
    published_date = parse_finnhub_datetime(timestamp)

    if published_date is None:
        return False

    cutoff = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)
    return published_date >= cutoff


def clean_ticker(ticker):
    ticker = str(ticker).upper().strip()
    ticker = ticker.replace("$", "")
    ticker = ticker.split(":")[-1]
    ticker = ticker.split(".")[0]
    return ticker


def clean_company_name(name):
    name = str(name).upper()
    remove_words = [
        " INC", " INC.", " CORPORATION", " CORP", " CORP.", " COMPANY", " CO.", " CO ",
        " LTD", " LTD.", " PLC", " SA", " AG", " NV", " HOLDINGS", " HOLDING", " GROUP",
        " CLASS A", " CLASS B", " COMMON STOCK", " ORDINARY SHARES",
    ]

    for word in remove_words:
        name = name.replace(word, " ")

    name = re.sub(r"[^A-Z0-9 ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def company_name_in_text(company_name, clean_text):
    if not company_name or len(company_name) < MIN_COMPANY_NAME_LENGTH:
        return False

    return f" {company_name} " in f" {clean_text} "


def load_us_symbols():
    if os.path.exists(SYMBOL_CACHE_FILE):
        symbols_df = pd.read_csv(SYMBOL_CACHE_FILE)
    else:
        url = "https://finnhub.io/api/v1/stock/symbol"
        params = {
            "exchange": "US",
            "token": FINNHUB_API_KEY,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            symbols_df = pd.DataFrame(response.json())
            symbols_df.to_csv(SYMBOL_CACHE_FILE, index=False)
        except requests.exceptions.RequestException as error:
            print("Failed to fetch Finnhub US symbols:", error)
            return pd.DataFrame()

    if symbols_df.empty or "symbol" not in symbols_df.columns:
        return pd.DataFrame()

    if "description" not in symbols_df.columns:
        symbols_df["description"] = ""

    if "type" not in symbols_df.columns:
        symbols_df["type"] = ""

    symbols_df["type"] = symbols_df["type"].astype(str).str.upper()
    symbols_df["symbol"] = symbols_df["symbol"].astype(str).str.upper()
    symbols_df["clean_name"] = symbols_df["description"].apply(clean_company_name)
    symbols_df["first_name_word"] = symbols_df["clean_name"].str.split().str[0].fillna("")
    symbols_df = symbols_df[
        symbols_df["symbol"].str.match(r"^[A-Z]{1,5}$", na=False)
        & ~symbols_df["symbol"].isin(COMMON_FALSE_TICKERS)
        & (symbols_df["clean_name"].str.len() >= MIN_COMPANY_NAME_LENGTH)
        & symbols_df["type"].str.contains("COMMON|ADR|REIT|ETP|ETF", regex=True, na=False)
    ]

    if EXCLUDE_OTC_STYLE_TICKERS:
        symbols_df = symbols_df[~((symbols_df["symbol"].str.len() == 5) & (symbols_df["symbol"].str.endswith(("F", "Y"))))]

    symbols_df = symbols_df[symbols_df["first_name_word"].str.len() >= MIN_NAME_WORD_LENGTH]

    return symbols_df[["symbol", "description", "clean_name", "first_name_word", "type"]].drop_duplicates("symbol")


def clean_ticker_list(tickers, valid_symbols):
    clean_tickers = []

    for ticker in tickers:
        if not ticker:
            continue

        if ticker not in valid_symbols:
            continue

        if ticker in COMMON_FALSE_TICKERS:
            continue

        if not ticker.isalpha() or not 1 <= len(ticker) <= 5:
            continue

        if EXCLUDE_OTC_STYLE_TICKERS and len(ticker) == 5 and ticker.endswith(("F", "Y")):
            continue

        clean_tickers.append(ticker)

    return sorted(set(clean_tickers))


def extract_ticker_matches(text, related=None, symbols_df=None):
    upper_text = text.upper()
    clean_text = clean_company_name(text)
    valid_symbols = set(symbols_df["symbol"]) if symbols_df is not None and not symbols_df.empty else set()
    matches = []

    cashtag_matches = re.findall(r"\$([A-Z]{1,5})\b", upper_text)
    for ticker in clean_ticker_list(cashtag_matches, valid_symbols):
        matches.append({
            "ticker": ticker,
            "confidence": "HIGH",
            "reason": "cashtag",
        })

    if symbols_df is not None and not symbols_df.empty:
        article_words = set(clean_text.split())
        possible_symbols = symbols_df[symbols_df["first_name_word"].isin(article_words)]

        for _, row in possible_symbols.iterrows():
            symbol = row["symbol"]
            company_name = row["clean_name"]

            if not company_name or len(company_name) < MIN_COMPANY_NAME_LENGTH:
                continue

            if company_name_in_text(company_name, clean_text):
                matches.append({
                    "ticker": symbol,
                    "confidence": "HIGH",
                    "reason": "company_name",
                })

    if related:
        related_tickers = []

        if isinstance(related, str):
            related_tickers = [clean_ticker(item) for item in related.split(",")]
        elif isinstance(related, list):
            related_tickers = [clean_ticker(item) for item in related]

        for ticker in clean_ticker_list(related_tickers, valid_symbols):
            already_high = any(match["ticker"] == ticker and match["confidence"] == "HIGH" for match in matches)

            if already_high:
                continue

            matches.append({
                "ticker": ticker,
                "confidence": "LOW",
                "reason": "related_only",
            })

    if MIN_TICKER_CONFIDENCE == "HIGH":
        matches = [match for match in matches if match["confidence"] == "HIGH"]

    deduped = {}
    for match in matches:
        ticker = match["ticker"]
        if ticker not in deduped or match["confidence"] == "HIGH":
            deduped[ticker] = match

    return list(deduped.values())


def extract_tickers(text, related=None, symbols_df=None):
    return sorted(match["ticker"] for match in extract_ticker_matches(text, related, symbols_df))



def has_tradeable_market_data(ticker):
    try:
        df = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=False, threads=False)
    except Exception:
        return False, None, None

    if df.empty or "Close" not in df.columns or "Volume" not in df.columns:
        return False, None, None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    last_price = float(df["Close"].dropna().iloc[-1]) if not df["Close"].dropna().empty else 0
    avg_volume = float(df["Volume"].dropna().tail(10).mean()) if not df["Volume"].dropna().empty else 0
    is_tradeable = last_price >= MIN_LAST_PRICE and avg_volume >= MIN_AVG_VOLUME

    return is_tradeable, last_price, avg_volume


def fetch_yahoo_screener_tickers():
    tickers = set()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    for scr_id in YAHOO_GAINERS_SCR_IDS:
        params = {
            "scrIds": scr_id,
            "count": YAHOO_GAINERS_COUNT,
        }

        try:
            response = requests.get(YAHOO_SCREENER_URL, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()
            result = data.get("finance", {}).get("result", [])

            for block in result:
                for quote in block.get("quotes", []):
                    symbol = clean_ticker(quote.get("symbol", ""))

                    if not symbol:
                        continue

                    if not symbol.isalpha() or len(symbol) > 5:
                        continue

                    if symbol in COMMON_FALSE_TICKERS:
                        continue

                    if EXCLUDE_OTC_STYLE_TICKERS and len(symbol) == 5 and symbol.endswith(("F", "Y")):
                        continue

                    tickers.add(symbol)

        except Exception as error:
            print(f"Yahoo screener failed for {scr_id}:", type(error).__name__)

    return sorted(tickers)

# --- Top Movers logic ---
def get_close_series(df, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        if ticker not in df.columns.get_level_values(0):
            return pd.Series(dtype=float)
        ticker_df = df[ticker]
        if "Close" not in ticker_df.columns:
            return pd.Series(dtype=float)
        return ticker_df["Close"].dropna()

    if "Close" not in df.columns:
        return pd.Series(dtype=float)

    return df["Close"].dropna()


def get_volume_series(df, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        if ticker not in df.columns.get_level_values(0):
            return pd.Series(dtype=float)
        ticker_df = df[ticker]
        if "Volume" not in ticker_df.columns:
            return pd.Series(dtype=float)
        return ticker_df["Volume"].dropna()

    if "Volume" not in df.columns:
        return pd.Series(dtype=float)

    return df["Volume"].dropna()


def scan_top_movers():
    if not ENABLE_MOVER_CANDIDATES:
        return pd.DataFrame()

    mover_tickers = fetch_yahoo_screener_tickers()
    print("Yahoo screener tickers loaded:", len(mover_tickers))

    if not mover_tickers:
        return pd.DataFrame()

    try:
        df = yf.download(
            mover_tickers,
            period="10d",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
            group_by="ticker",
        )
    except Exception as error:
        print("Failed to scan top movers:", error)
        return pd.DataFrame()

    mover_rows = []

    for ticker in mover_tickers:
        close = get_close_series(df, ticker)
        volume = get_volume_series(df, ticker)

        if len(close) < 6 or volume.empty:
            continue

        last_price = float(close.iloc[-1])
        yesterday_return = (close.iloc[-1] / close.iloc[-2]) - 1
        week_return = (close.iloc[-1] / close.iloc[-6]) - 1
        avg_volume = float(volume.tail(10).mean())

        if last_price < MIN_LAST_PRICE or avg_volume < MIN_AVG_VOLUME:
            continue

        if yesterday_return < MIN_YESTERDAY_RETURN and week_return < MIN_WEEK_RETURN:
            continue

        mover_rows.append({
            "ticker": ticker,
            "article_count": 0,
            "positive_articles": 0,
            "negative_articles": 0,
            "news_score": 0,
            "latest_headline": f"Yahoo top mover: yesterday {yesterday_return * 100:.2f}%, week {week_return * 100:.2f}%",
            "latest_url": "",
            "gemini_sentiment": "MOMENTUM",
            "gemini_confidence": 100,
            "gemini_reason": "Added from Yahoo Finance mover screener because the stock was a top recent price mover.",
            "match_confidence": "HIGH",
            "match_reason": "yahoo_price_momentum",
            "last_price": last_price,
            "avg_volume": avg_volume,
            "candidate_source": "MOVER",
            "yesterday_return_pct": round(yesterday_return * 100, 2),
            "week_return_pct": round(week_return * 100, 2),
            "status": "NEWS_CANDIDATE",
        })

    if not mover_rows:
        return pd.DataFrame()

    movers_df = pd.DataFrame(mover_rows)
    movers_df = movers_df.sort_values(
        ["week_return_pct", "yesterday_return_pct", "avg_volume"],
        ascending=[False, False, False],
    ).head(TOP_MOVER_LIMIT)

    movers_df.to_csv(os.path.join(DATA_DIR, "top_movers.csv"), index=False)
    return movers_df


def fetch_finnhub_market_news():
    if not FINNHUB_API_KEY:
        print("Missing FINNHUB_API_KEY environment variable.")
        print("Put it in .env like: FINNHUB_API_KEY=\"your_key_here\"")
        raise SystemExit

    url = "https://finnhub.io/api/v1/news"
    params = {
        "category": CATEGORY,
        "token": FINNHUB_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as error:
        print("Finnhub request failed:", error)
        print("Response:", response.text)
        return []
    except requests.exceptions.RequestException as error:
        print("Finnhub request failed:", error)
        return []


def normalize_finnhub_article(article, symbols_df):
    headline = article.get("headline", "") or ""
    summary = article.get("summary", "") or ""
    source = article.get("source", "") or ""
    url = article.get("url", "") or ""
    published_timestamp = article.get("datetime")
    related = article.get("related", "") or ""
    text = f"{headline} {summary}"
    published_date = parse_finnhub_datetime(published_timestamp)

    ticker_matches = extract_ticker_matches(text, related, symbols_df)

    return {
        "headline": headline,
        "summary": summary[:300],
        "source": source,
        "url": url,
        "published": published_date.isoformat() if published_date else "",
        "published_timestamp": published_timestamp,
        "tickers": sorted(match["ticker"] for match in ticker_matches),
        "ticker_matches": ticker_matches,
        "article_score": score_text(text),
    }


def scan_market_news():
    articles = fetch_finnhub_market_news()
    symbols_df = load_us_symbols()
    ticker_rows = []
    debug_rows = []
    market_data_cache = {}

    print("Raw articles returned:", len(articles))
    print("US symbols loaded:", len(symbols_df))

    for raw_article in articles:
        article = normalize_finnhub_article(raw_article, symbols_df)
        recent = is_recent_article(article["published_timestamp"])
        has_tickers = bool(article["tickers"])

        debug_rows.append({
            "headline": article["headline"],
            "published": article["published"],
            "recent": recent,
            "tickers": ", ".join(article["tickers"]),
            "match_reasons": ", ".join(f"{match['ticker']}:{match['reason']}:{match['confidence']}" for match in article["ticker_matches"]),
            "article_score": article["article_score"],
            "url": article["url"],
        })

        if not recent:
            continue

        if not has_tickers:
            continue

        for ticker in article["tickers"]:
            if ticker not in market_data_cache:
                market_data_cache[ticker] = has_tradeable_market_data(ticker)

            is_tradeable, last_price, avg_volume = market_data_cache[ticker]

            if not is_tradeable:
                continue

            match = next((item for item in article["ticker_matches"] if item["ticker"] == ticker), None)

            gemini_sentiment, gemini_confidence, gemini_reason = classify_with_gemini(
                ticker,
                article["headline"],
                article["summary"],
            )

            ticker_rows.append({
                "ticker": ticker,
                "headline": article["headline"],
                "summary": article["summary"],
                "source": article["source"],
                "url": article["url"],
                "published": article["published"],
                "article_score": article["article_score"],
                "gemini_sentiment": gemini_sentiment,
                "gemini_confidence": gemini_confidence,
                "gemini_reason": gemini_reason,
                "match_confidence": match["confidence"] if match else "",
                "match_reason": match["reason"] if match else "",
                "last_price": last_price,
                "avg_volume": avg_volume,
            })

    pd.DataFrame(debug_rows).to_csv(NEWS_DEBUG_FILE, index=False)

    if not ticker_rows:
        return pd.DataFrame(), pd.DataFrame(debug_rows)

    ticker_news = pd.DataFrame(ticker_rows)

    summary = (
        ticker_news.groupby("ticker")
        .agg(
            article_count=("headline", "count"),
            positive_articles=("article_score", lambda x: (x > 0).sum()),
            negative_articles=("article_score", lambda x: (x < 0).sum()),
            news_score=("article_score", "sum"),
            latest_headline=("headline", "first"),
            latest_url=("url", "first"),
            gemini_sentiment=("gemini_sentiment", "first"),
            gemini_confidence=("gemini_confidence", "first"),
            gemini_reason=("gemini_reason", "first"),
            match_confidence=("match_confidence", "first"),
            match_reason=("match_reason", "first"),
            last_price=("last_price", "first"),
            avg_volume=("avg_volume", "first"),
        )
        .reset_index()
    )

    summary["candidate_source"] = "NEWS"
    summary["yesterday_return_pct"] = None
    summary["week_return_pct"] = None

    summary["status"] = summary.apply(
        lambda row: "NEWS_CANDIDATE"
        if row["article_count"] >= MIN_ARTICLES
        and row["gemini_sentiment"] == "POSITIVE"
        and row["gemini_confidence"] >= MIN_GEMINI_CONFIDENCE
        else "SKIP",
        axis=1,
    )

    summary = summary.sort_values(
        ["status", "news_score", "positive_articles", "article_count"],
        ascending=[True, False, False, False],
    )

    return summary, ticker_news


def main():
    print("Running file:", os.path.abspath(__file__))
    print("News lookback days:", DAYS_BACK)
    if USE_GEMINI_SENTIMENT and not GEMINI_API_KEY:
        print("Missing GEMINI_API_KEY in .env")
        raise SystemExit

    if USE_GEMINI_SENTIMENT:
        models = get_available_gemini_models()
        print("Gemini models available:", models[:5])

    summary_df, articles_df = scan_market_news()
    movers_df = scan_top_movers()

    if not movers_df.empty:
        print("\nTop mover candidates")
        print(movers_df[["ticker", "yesterday_return_pct", "week_return_pct", "last_price", "avg_volume"]].to_string(index=False))

    if summary_df.empty and movers_df.empty:
        print("No ticker candidates found from recent Finnhub news or top movers.")
        print(f"Saved debug details to {NEWS_DEBUG_FILE}")
        return

    if summary_df.empty:
        summary_df = movers_df
    elif not movers_df.empty:
        summary_df = pd.concat([summary_df, movers_df], ignore_index=True)
        summary_df = summary_df.sort_values(
            ["status", "candidate_source", "gemini_confidence", "avg_volume"],
            ascending=[True, True, False, False],
        ).drop_duplicates("ticker", keep="first")

    candidates = summary_df[summary_df["status"] == "NEWS_CANDIDATE"]

    summary_df.to_csv(NEWS_CANDIDATES_FILE, index=False)
    articles_df.to_csv(NEWS_ARTICLES_FILE, index=False)

    print("\nNews Candidate Summary")
    print(summary_df.to_string(index=False))

    print("\nTickers to send to pattern bot")
    if candidates.empty:
        print("No strong news candidates found today.")
    else:
        print(candidates["ticker"].tolist())

    print(f"\nSaved summary to {NEWS_CANDIDATES_FILE}")
    print(f"Saved article details to {NEWS_ARTICLES_FILE}")
    print(f"Saved debug details to {NEWS_DEBUG_FILE}")


if __name__ == "__main__":
    main()
