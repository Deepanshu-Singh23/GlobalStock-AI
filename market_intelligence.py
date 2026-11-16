
import os
import re
import numpy as np
import pandas as pd


def normalize_symbol(symbol):
    return re.sub(
        r"[^A-Z0-9]",
        "_",
        str(symbol).upper()
    ).strip("_")


def find_column(df, candidates):
    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def load_price_file(path):
    try:
        df = pd.read_csv(path)

        if df.empty:
            return pd.DataFrame()

        date_col = find_column(
            df, ["date", "datetime", "timestamp"]
        )

        close_col = find_column(
            df, ["close", "adj close", "adj_close"]
        )

        if date_col is None or close_col is None:
            return pd.DataFrame()

        prices = pd.DataFrame({
            "date": pd.to_datetime(
                df[date_col], errors="coerce"
            ),
            "close": pd.to_numeric(
                df[close_col], errors="coerce"
            )
        })

        prices = (
            prices.dropna()
            .sort_values("date")
            .drop_duplicates("date")
        )

        prices = prices[prices["close"] > 0]

        return prices.reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


def calculate_stock_metrics(prices):

    if prices.empty or len(prices) < 2:
        return {}

    latest = prices.iloc[-1]
    previous = prices.iloc[-2]

    latest_date = latest["date"]
    latest_close = float(latest["close"])

    metrics = {
        "latest_close": latest_close,
        "price_date": latest_date,
        "daily_return_pct": (
            (latest_close / previous["close"] - 1) * 100
        )
    }

    windows = {
        "return_1m_pct": (30, 7),
        "return_3m_pct": (90, 7),
        "return_1y_pct": (365, 14)
    }

    for metric, (days, tolerance) in windows.items():

        target_date = latest_date - pd.Timedelta(days=days)

        eligible = prices[
            prices["date"] <= target_date
        ]

        if eligible.empty:
            metrics[metric] = np.nan
            continue

        reference = eligible.iloc[-1]

        gap = (target_date - reference["date"]).days

        if gap > tolerance or reference["close"] <= 0:
            metrics[metric] = np.nan
        else:
            metrics[metric] = (
                (latest_close / reference["close"] - 1) * 100
            )

    return metrics


def build_market_dataset(master_file, price_path):

    stocks = pd.read_csv(master_file)

    required = [
        "symbol",
        "company_name",
        "country",
        "exchange_code"
    ]

    missing = [
        col for col in required
        if col not in stocks.columns
    ]

    if missing:
        raise ValueError(
            f"Master dataset missing columns: {missing}"
        )

    stocks["file_key"] = (
        stocks["symbol"].astype(str).map(normalize_symbol)
    )

    price_files = {}

    for filename in os.listdir(price_path):

        if filename.lower().endswith(".csv"):

            key = normalize_symbol(
                os.path.splitext(filename)[0]
            )

            price_files[key] = os.path.join(
                price_path, filename
            )

    records = []

    for _, stock in stocks.iterrows():

        key = stock["file_key"]

        if key not in price_files:
            continue

        prices = load_price_file(price_files[key])
        metrics = calculate_stock_metrics(prices)

        if not metrics:
            continue

        record = stock.to_dict()
        record.update(metrics)

        records.append(record)

    result = pd.DataFrame(records)

    if not result.empty:
        result = result.drop_duplicates(
            subset=["symbol"]
        )

    return result


def summarize_market(data, group_column):

    if (
        data.empty
        or group_column not in data.columns
    ):
        return pd.DataFrame()

    summary = (
        data.groupby(group_column, dropna=True)
        .agg(
            stocks_with_prices=("symbol", "nunique"),
            median_daily_return=(
                "daily_return_pct", "median"
            ),
            advancing_stocks=(
                "daily_return_pct",
                lambda x: int((x > 0).sum())
            ),
            declining_stocks=(
                "daily_return_pct",
                lambda x: int((x < 0).sum())
            ),
            unchanged_stocks=(
                "daily_return_pct",
                lambda x: int((x == 0).sum())
            ),
            median_1m_return=(
                "return_1m_pct", "median"
            ),
            median_3m_return=(
                "return_3m_pct", "median"
            ),
            median_1y_return=(
                "return_1y_pct", "median"
            )
        )
        .reset_index()
    )

    return summary.sort_values(
        "stocks_with_prices",
        ascending=False
    )


def get_market_overview(data):

    if data.empty:
        return {
            "stocks_with_prices": 0,
            "countries": 0,
            "exchanges": 0,
            "advancing": 0,
            "declining": 0,
            "unchanged": 0
        }

    daily = data["daily_return_pct"].dropna()

    return {
        "stocks_with_prices": int(
            data["symbol"].nunique()
        ),
        "countries": int(
            data["country"].nunique()
        ),
        "exchanges": int(
            data["exchange_code"].nunique()
        ),
        "advancing": int((daily > 0).sum()),
        "declining": int((daily < 0).sum()),
        "unchanged": int((daily == 0).sum())
    }


def get_sector_coverage(master_file):

    stocks = pd.read_csv(master_file)

    if "sector" not in stocks.columns:
        return {
            "total": len(stocks),
            "available": 0,
            "missing": len(stocks),
            "coverage_pct": 0.0
        }

    available = int(stocks["sector"].notna().sum())
    total = len(stocks)

    return {
        "total": total,
        "available": available,
        "missing": total - available,
        "coverage_pct": (
            available / total * 100 if total else 0
        )
    }


print("Full Market Intelligence engine loaded!")
