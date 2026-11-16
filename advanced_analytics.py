
import os
import re
import glob
import numpy as np
import pandas as pd


def normalize_symbol(value):
    return re.sub(
        r"[^A-Z0-9_]",
        "_",
        str(value).strip().upper()
    )


def list_available_stocks(master_path, price_path):

    master = pd.read_csv(master_path)

    required = [
        "symbol",
        "company_name",
        "country",
        "exchange_code"
    ]

    for col in required:
        if col not in master.columns:
            raise ValueError(f"Missing master column: {col}")

    master["file_key"] = master["symbol"].map(
        normalize_symbol
    )

    files = {
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(
            os.path.join(price_path, "*.csv")
        )
        if os.path.getsize(path) > 100
    }

    available = master[
        master["file_key"].isin(files)
    ].copy()

    return (
        available
        .drop_duplicates("file_key")
        .sort_values(["country", "symbol"])
        .reset_index(drop=True)
    )


def load_close_prices(file_keys, price_path):

    series_list = []

    for key in file_keys:

        path = os.path.join(
            price_path,
            str(key) + ".csv"
        )

        if not os.path.exists(path):
            continue

        try:
            df = pd.read_csv(path)

            date_col = next(
                (c for c in ["date", "Date", "Datetime"]
                 if c in df.columns),
                None
            )

            price_col = next(
                (c for c in ["adj_close", "Adj Close",
                             "close", "Close"]
                 if c in df.columns),
                None
            )

            if date_col is None or price_col is None:
                continue

            dates = pd.to_datetime(
                df[date_col],
                errors="coerce",
                utc=True
            ).dt.tz_convert(None)

            values = pd.to_numeric(
                df[price_col],
                errors="coerce"
            )

            temp = pd.DataFrame({
                "date": dates,
                "price": values
            })

            temp = (
                temp.dropna()
                .drop_duplicates("date")
                .sort_values("date")
            )

            if len(temp) < 20:
                continue

            s = temp.set_index("date")["price"]
            s.name = str(key)

            series_list.append(s)

        except Exception:
            continue

    if not series_list:
        return pd.DataFrame()

    return pd.concat(
        series_list,
        axis=1
    ).sort_index()


def calculate_risk_metrics(prices):

    results = []

    if prices.empty:
        return pd.DataFrame()

    for symbol in prices.columns:

        series = prices[symbol].dropna()

        if len(series) < 2:
            continue

        returns = series.pct_change().dropna()

        running_max = series.cummax()

        drawdown = (
            (series - running_max) / running_max
        ) * 100

        total_return = (
            (series.iloc[-1] / series.iloc[0]) - 1
        ) * 100

        volatility = (
            returns.std() * np.sqrt(252) * 100
        )

        results.append({
            "file_key": symbol,
            "Observations": len(series),
            "Start Date": series.index.min(),
            "End Date": series.index.max(),
            "Historical Return (%)": round(total_return, 2),
            "Annualized Volatility (%)": round(volatility, 2),
            "Maximum Drawdown (%)": round(drawdown.min(), 2),
            "Average Daily Return (%)": round(
                returns.mean() * 100, 4
            )
        })

    return pd.DataFrame(results)


def calculate_correlation(prices):

    returns = prices.pct_change()

    return returns.corr(
        min_periods=20
    )


def normalized_performance(prices):

    normalized = pd.DataFrame(
        index=prices.index
    )

    for col in prices.columns:

        series = prices[col].dropna()

        if series.empty or series.iloc[0] == 0:
            continue

        normalized[col] = (
            prices[col] / series.iloc[0]
        ) * 100

    return normalized


def rolling_volatility(prices, window=20):

    returns = prices.pct_change()

    return (
        returns
        .rolling(window)
        .std()
        * np.sqrt(252)
        * 100
    )


def save_analytics(metrics, correlation, normalized):

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.dirname(__file__)
        )),
        "data",
        "processed",
        "advanced_analytics"
    )

    os.makedirs(output_path, exist_ok=True)

    metrics.to_csv(
        os.path.join(output_path, "risk_metrics.csv"),
        index=False
    )

    correlation.to_csv(
        os.path.join(output_path, "correlation_matrix.csv")
    )

    normalized.to_csv(
        os.path.join(output_path, "normalized_performance.csv")
    )

    return output_path
