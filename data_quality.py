
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def audit_dataframe(df):
    if df.empty:
        return {
            "rows": 0,
            "columns": 0,
            "duplicate_rows": 0,
            "missing_cells": 0,
            "missing_pct": 0
        }

    total = df.size
    missing = int(df.isna().sum().sum())

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": missing,
        "missing_pct": round(missing / total * 100, 2) if total else 0
    }

def column_quality(df):
    if df.empty:
        return pd.DataFrame()

    result = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "available_count": df.notna().sum().values,
        "missing_pct": (df.isna().mean().values * 100).round(2),
        "data_type": df.dtypes.astype(str).values
    })

    return result.sort_values("missing_pct", ascending=False)

def duplicate_symbols(df):
    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame()

    keys = ["symbol"]
    if "country" in df.columns:
        keys = ["country", "symbol"]

    return df[df.duplicated(keys, keep=False)].sort_values(keys)

def price_file_audit(price_path):
    files = glob.glob(os.path.join(price_path, "*.csv"))
    records = []

    for path in files:
        try:
            df = pd.read_csv(path)
            date_col = next(
                (c for c in df.columns if str(c).lower() in ["date", "datetime"]),
                None
            )
            close_col = next(
                (c for c in df.columns if str(c).lower() in ["close", "adj close", "adj_close"]),
                None
            )

            latest_date = None
            if date_col:
                dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
                if not dates.empty:
                    latest_date = dates.max().date()

            records.append({
                "file": os.path.basename(path),
                "rows": len(df),
                "close_column_available": close_col is not None,
                "date_column_available": date_col is not None,
                "latest_date": str(latest_date) if latest_date else None,
                "missing_close": (
                    int(pd.to_numeric(df[close_col], errors="coerce").isna().sum())
                    if close_col else None
                )
            })
        except Exception as error:
            records.append({
                "file": os.path.basename(path),
                "rows": 0,
                "close_column_available": False,
                "date_column_available": False,
                "latest_date": None,
                "missing_close": None,
                "error": str(error)
            })

    return pd.DataFrame(records)

def coverage_table(master, market, group_col):
    if master.empty or market.empty or group_col not in master.columns:
        return pd.DataFrame()

    master_counts = master.groupby(group_col)["symbol"].nunique()
    market_counts = market.groupby(group_col)["symbol"].nunique()

    result = pd.concat(
        [master_counts.rename("master_listings"),
         market_counts.rename("with_prices")],
        axis=1
    ).fillna(0)

    result["master_listings"] = result["master_listings"].astype(int)
    result["with_prices"] = result["with_prices"].astype(int)
    result["without_prices"] = result["master_listings"] - result["with_prices"]
    result["coverage_pct"] = (
        result["with_prices"] / result["master_listings"].replace(0, np.nan) * 100
    ).round(2)

    return result.reset_index()

def date_audit(market):
    if market.empty or "price_date" not in market.columns:
        return pd.DataFrame()

    result = market.copy()
    result["price_date"] = pd.to_datetime(result["price_date"], errors="coerce")

    return (
        result.groupby("price_date")
        .agg(
            securities=("symbol", "nunique"),
            countries=("country", "nunique")
        )
        .reset_index()
        .sort_values("price_date", ascending=False)
    )

def quality_score(market, master):
    if market.empty:
        return 0

    completeness = market.notna().mean().mean() * 100
    duplicate_penalty = min(
        market.duplicated(["country", "symbol"]).sum() / max(len(market), 1) * 100,
        20
    )

    coverage = min(len(market) / max(len(master), 1) * 100, 100)

    return round(max(0, 0.5 * completeness + 0.5 * coverage - duplicate_penalty), 2)

def build_quality_report(master, market, price_path):
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "master_audit": audit_dataframe(master),
        "market_audit": audit_dataframe(market),
        "price_files": price_file_audit(price_path),
        "column_quality": column_quality(market),
        "duplicate_symbols": duplicate_symbols(market),
        "country_coverage": coverage_table(master, market, "country"),
        "exchange_coverage": coverage_table(master, market, "exchange_code"),
        "date_audit": date_audit(market),
        "quality_score": quality_score(market, master)
    }

print("Data Quality engine created successfully.")
