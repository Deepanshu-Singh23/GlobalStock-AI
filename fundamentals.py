
import os
import json
import time
import tempfile
import yfinance as yf

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

FUNDAMENTALS_PATH = os.path.join(
    tempfile.gettempdir(),
    "globalstock_ai",
    "fundamentals"
)

os.makedirs(FUNDAMENTALS_PATH, exist_ok=True)

FUNDAMENTAL_FIELDS = [
    "longName", "shortName", "symbol",
    "quoteType", "sector", "industry", "country", "currency",
    "marketCap", "enterpriseValue",
    "trailingPE", "forwardPE", "priceToBook",
    "trailingEps", "forwardEps",
    "totalRevenue", "revenueGrowth",
    "grossMargins", "operatingMargins",
    "profitMargins", "netIncomeToCommon",
    "dividendYield", "beta", "website",
    "longBusinessSummary", "fullTimeEmployees",
    "currentPrice", "previousClose",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "fundFamily", "category",
    "navPrice", "netAssets", "totalAssets",
    "netExpenseRatio", "annualReportExpenseRatio",
    "yield", "trailingAnnualDividendYield",
    "ytdReturn", "trailingThreeMonthReturns",
    "trailingThreeMonthNavReturns",
    "beta3Year", "regularMarketPrice",
    "regularMarketVolume", "averageVolume",
    "averageDailyVolume10Day",
    "averageDailyVolume3Month",
    "fundInceptionDate"
]


def get_cache_file(symbol):
    safe_symbol = (
        str(symbol).replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    return os.path.join(
        FUNDAMENTALS_PATH,
        safe_symbol + ".json"
    )


def get_fundamentals(symbol, force_refresh=False):

    cache_file = get_cache_file(symbol)

    if os.path.exists(cache_file) and not force_refresh:
        age = time.time() - os.path.getmtime(cache_file)

        if age < 86400:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f), "Loaded from local cache"
            except (OSError, ValueError):
                pass

    try:
        ticker = yf.Ticker(symbol)
        raw_info = ticker.get_info()

        if not raw_info:
            return {}, "No fundamentals returned by Yahoo Finance"

        info = {
            key: raw_info.get(key)
            for key in FUNDAMENTAL_FIELDS
        }

        info["data_source"] = "Yahoo Finance"
        info["retrieved_at"] = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        try:
            with open(cache_file, "w") as f:
                json.dump(info, f, indent=4, default=str)
        except OSError:
            pass

        return info, "Fresh fundamentals downloaded"

    except Exception as error:
        return {}, f"Fundamentals error: {str(error)}"
