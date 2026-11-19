
import os
import time
import tempfile
import pandas as pd
import yfinance as yf

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

FINANCIALS_PATH = os.path.join(
    tempfile.gettempdir(),
    "globalstock_ai",
    "financials"
)

os.makedirs(FINANCIALS_PATH, exist_ok=True)


def get_financial_file(symbol, period_type):
    safe_symbol = (
        str(symbol).replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    return os.path.join(
        FINANCIALS_PATH,
        f"{safe_symbol}_{period_type}.csv"
    )


def extract_metric(statement, aliases):
    if statement is None or statement.empty:
        return None

    for index in statement.index:
        label = str(index).strip().lower()

        for alias in aliases:
            if alias in label:
                return statement.loc[index]

    return None


def prepare_statement(statement, period_type):

    if statement is None or statement.empty:
        return pd.DataFrame()

    revenue = extract_metric(
        statement,
        ["total revenue", "operating revenue"]
    )

    net_income = extract_metric(
        statement,
        ["net income"]
    )

    operating_income = extract_metric(
        statement,
        ["operating income"]
    )

    gross_profit = extract_metric(
        statement,
        ["gross profit"]
    )

    records = []

    for date in statement.columns:

        def get_value(series):
            if series is None:
                return None

            try:
                value = series[date]
                return float(value) if pd.notna(value) else None
            except Exception:
                return None

        records.append({
            "period": pd.to_datetime(date).strftime("%Y-%m-%d"),
            "period_type": period_type,
            "total_revenue": get_value(revenue),
            "net_income": get_value(net_income),
            "operating_income": get_value(operating_income),
            "gross_profit": get_value(gross_profit)
        })

    result = pd.DataFrame(records)

    if not result.empty:
        result = result.sort_values("period")

    return result


def get_financial_statements(symbol, force_refresh=False):

    annual_file = get_financial_file(symbol, "annual")
    quarterly_file = get_financial_file(symbol, "quarterly")

    cache_files = [annual_file, quarterly_file]

    if not force_refresh and all(
        os.path.exists(file) for file in cache_files
    ):
        ages = [
            time.time() - os.path.getmtime(file)
            for file in cache_files
        ]

        if all(age < 86400 for age in ages):
            try:
                annual = pd.read_csv(annual_file)
                quarterly = pd.read_csv(quarterly_file)

                return (
                    annual,
                    quarterly,
                    "Loaded from local cache"
                )
            except Exception:
                pass

    try:
        ticker = yf.Ticker(symbol)

        annual_raw = ticker.income_stmt
        quarterly_raw = ticker.quarterly_income_stmt

        annual = prepare_statement(
            annual_raw, "Annual"
        )

        quarterly = prepare_statement(
            quarterly_raw, "Quarterly"
        )

        if annual.empty and quarterly.empty:
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                "No income statement data available"
            )

        annual.to_csv(annual_file, index=False)
        quarterly.to_csv(quarterly_file, index=False)

        return (
            annual,
            quarterly,
            "Fresh financial statements downloaded and saved"
        )

    except Exception as error:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            f"Financial data error: {str(error)}"
        )
