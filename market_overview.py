
import os
import re
import glob
import numpy as np
import pandas as pd

PROJECT_PATH = "/content/drive/MyDrive/GlobalStock_AI"

MASTER_PATH = os.path.join(
    PROJECT_PATH, "data", "processed",
    "global_stock_master.csv"
)

PRICE_PATH = os.path.join(
    PROJECT_PATH, "data", "raw", "prices"
)

def load_market_master():
    if not os.path.exists(MASTER_PATH):
        return pd.DataFrame()

    return pd.read_csv(MASTER_PATH)

def normalize_symbol(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())

def get_market_summary():
    master = load_market_master()

    if master.empty:
        return {}

    price_files = glob.glob(
        os.path.join(PRICE_PATH, "*.csv")
    )

    price_keys = set()

    for path in price_files:
        filename = os.path.basename(path)
        filename = filename.replace("\\", "")
        filename = filename[:-4] if filename.lower().endswith(".csv") else filename
        price_keys.add(normalize_symbol(filename))

    symbols = master["symbol"].dropna().astype(str)
    matched = symbols.map(normalize_symbol).isin(price_keys)

    countries = master["country"].nunique()
    exchanges = master["exchange_code"].nunique()

    return {
        "total_securities": len(master),
        "countries": int(countries),
        "exchanges": int(exchanges),
        "price_files": len(price_files),
        "matched_securities": int(matched.sum()),
        "price_coverage": round(
            matched.sum() / len(master) * 100, 2
        ) if len(master) else 0
    }

def get_country_distribution():
    master = load_market_master()

    if master.empty or "country" not in master.columns:
        return pd.DataFrame()

    return (
        master.groupby("country")
        .size()
        .reset_index(name="Securities")
        .sort_values("Securities", ascending=False)
    )

def get_exchange_distribution():
    master = load_market_master()

    if master.empty or "exchange_code" not in master.columns:
        return pd.DataFrame()

    return (
        master.groupby("exchange_code")
        .size()
        .reset_index(name="Securities")
        .sort_values("Securities", ascending=False)
    )

def get_market_movers():
    master = load_market_master()

    if master.empty:
        return pd.DataFrame()

    symbol_lookup = {}

    for _, row in master.iterrows():
        symbol = str(row.get("symbol", "")).strip()
        if symbol:
            symbol_lookup[normalize_symbol(symbol)] = row.to_dict()

    results = []

    for path in glob.glob(os.path.join(PRICE_PATH, "*.csv")):
        filename = os.path.basename(path).replace("\\", "")
        filename = filename[:-4] if filename.lower().endswith(".csv") else filename

        key = normalize_symbol(filename)
        info = symbol_lookup.get(key)

        if info is None:
            continue

        try:
            data = pd.read_csv(path)

            price_column = (
                "adj_close" if "adj_close" in data.columns
                else "close" if "close" in data.columns
                else None
            )

            if price_column is None or "date" not in data.columns:
                continue

            data["date"] = pd.to_datetime(
                data["date"], errors="coerce"
            )

            data[price_column] = pd.to_numeric(
                data[price_column], errors="coerce"
            )

            data = data.dropna(
                subset=["date", price_column]
            ).sort_values("date")

            data = data[data[price_column] > 0]

            if len(data) < 2:
                continue

            previous = float(data.iloc[-2][price_column])
            latest = float(data.iloc[-1][price_column])

            if previous <= 0:
                continue

            change = ((latest - previous) / previous) * 100

            results.append({
                "Symbol": info.get("symbol"),
                "Company": info.get("company_name"),
                "Country": info.get("country"),
                "Exchange": info.get("exchange_code"),
                "Latest Price": latest,
                "Change (%)": change,
                "Latest Data Date": data.iloc[-1]["date"].strftime("%Y-%m-%d")
            })

        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(
        "Change (%)", ascending=False
    ).reset_index(drop=True)
