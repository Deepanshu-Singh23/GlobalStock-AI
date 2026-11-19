import os
import tempfile

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

FINANCIALS_PATH = os.path.join(
    tempfile.gettempdir(),
    "globalstock_ai",
    "financials"
)

os.makedirs(FINANCIALS_PATH, exist_ok=True)

FUNDAMENTAL_FIELDS = [
    # Company information
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

    # Fund / ETF information
    "fundFamily", "category",
    "navPrice", "netAssets", "totalAssets",
    "netExpenseRatio", "annualReportExpenseRatio",
    "yield", "trailingAnnualDividendYield",
    "ytdReturn", "trailingThreeMonthReturns",
    "trailingThreeMonthNavReturns",
    "beta3Year",
    "regularMarketPrice",
    "regularMarketVolume",
    "averageVolume",
    "averageDailyVolume10Day",
    "averageDailyVolume3Month",
    "fundInceptionDate"
]

def get_cache_file(symbol):
    safe_symbol = (
        symbol.replace(".", "_")
              .replace("/", "_")
              .replace("\\", "_")
    )
    return os.path.join(
        FUNDAMENTALS_PATH, safe_symbol + ".json"
    )

def get_fundamentals(symbol, force_refresh=False):

    cache_file = get_cache_file(symbol)

    # Reuse data if cache is less than 24 hours old
    if os.path.exists(cache_file) and not force_refresh:
        age = time.time() - os.path.getmtime(cache_file)

        if age < 86400:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f), "Loaded from Drive cache"
            except Exception:
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

        with open(cache_file, "w") as f:
            json.dump(info, f, indent=4, default=str)

        return info, "Fresh fundamentals downloaded and saved"

    except Exception as error:
        return {}, f"Fundamentals error: {str(error)}"
