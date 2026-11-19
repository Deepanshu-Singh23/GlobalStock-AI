
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
                (c for c in df.columns
                 if str(c).lower() in ["date", "datetime"]),
                None
            )

            close_col = next(
                (c for c in df.columns
                 if str(c).lower() in ["close", "adj close", "adj_close"]),
                None
            )

            dates = (
                pd.to_datetime(df[date_col], errors="coerce").dropna()
                if date_col else pd.Series(dtype="datetime64[ns]")
            )

            duplicate_dates = (
                int(dates.duplicated().sum()) if date_col else 0
            )

            missing_dates = 0
            if not dates.empty:
                expected = pd.bdate_range(dates.min(), dates.max())
                missing_dates = max(0, len(expected) - dates.nunique())

            missing_values = int(df.isna().sum().sum())

            invalid_ohlc = 0
            if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
                invalid_ohlc = int(
                    (
                        (df["High"] < df["Low"]) |
                        (df["High"] < df["Open"]) |
                        (df["High"] < df["Close"]) |
                        (df["Low"] > df["Open"]) |
                        (df["Low"] > df["Close"])
                    ).sum()
                )

            negative_volume = (
                int((pd.to_numeric(df["Volume"], errors="coerce") < 0).sum())
                if "Volume" in df.columns else 0
            )

            score = round(
                max(0, 100 - (
                    (missing_values / max(df.size, 1) * 40)
                    + duplicate_dates * 2
                    + invalid_ohlc * 5
                    + negative_volume * 5
                )), 2
            )

            records.append({
                "file": os.path.basename(path),
                "rows": len(df),
                "status": "OK" if close_col and date_col else "WARNING",
                "quality_score": score,
                "missing_values": missing_values,
                "duplicate_dates": duplicate_dates,
                "invalid_ohlc": invalid_ohlc,
                "negative_volume": negative_volume,
                "missing_dates": missing_dates,
                "latest_date": str(dates.max().date()) if not dates.empty else None
            })

        except Exception as error:
            records.append({
                "file": os.path.basename(path),
                "rows": 0,
                "status": "ERROR",
                "quality_score": 0,
                "missing_values": 0,
                "duplicate_dates": 0,
                "invalid_ohlc": 0,
                "negative_volume": 0,
                "missing_dates": 0,
                "error": str(error)
            })

    return pd.DataFrame(records)


def coverage_table(master, market, group_col):
    if master.empty or market.empty:
        return pd.DataFrame()

    if group_col not in master.columns or group_col not in market.columns:
        return pd.DataFrame()

    if "symbol" not in master.columns or "symbol" not in market.columns:
        return pd.DataFrame()

    master_counts = master.groupby(group_col)["symbol"].nunique()
    market_counts = market.groupby(group_col)["symbol"].nunique()

    result = pd.concat([
        master_counts.rename("master_listings"),
        market_counts.rename("with_prices")
    ], axis=1).fillna(0)

    result["master_listings"] = result["master_listings"].astype(int)
    result["with_prices"] = result["with_prices"].astype(int)
    result["without_prices"] = (
        result["master_listings"] - result["with_prices"]
    )

    result["coverage_pct"] = (
        result["with_prices"] /
        result["master_listings"].replace(0, np.nan) * 100
    ).round(2)

    return result.reset_index()


def build_coverage_report(master, market):
    return {
        "country": coverage_table(master, market, "country"),
        "exchange": coverage_table(master, market, "exchange_code")
    }


def build_group_quality(df, group_col):
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()

    return (
        df.groupby(group_col)
        .apply(
            lambda group: pd.Series({
                "total_records": len(group),
                "missing_values": int(group.isna().sum().sum()),
                "duplicate_rows": int(group.duplicated().sum()),
                "quality_score": round(
                    group.notna().mean().mean() * 100, 2
                )
            }),
            include_groups=False
        )
        .reset_index()
    )


def date_audit(market):
    if market.empty or "price_date" not in market.columns:
        return pd.DataFrame()

    result = market.copy()
    result["price_date"] = pd.to_datetime(
        result["price_date"], errors="coerce"
    )

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
        market.duplicated(["country", "symbol"]).sum()
        / max(len(market), 1) * 100,
        20
    ) if all(c in market.columns for c in ["country", "symbol"]) else 0

    coverage = min(
        len(market) / max(len(master), 1) * 100,
        100
    )

    return round(
        max(0, 0.5 * completeness + 0.5 * coverage - duplicate_penalty),
        2
    )


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


print("GlobalStock AI Data Quality Engine loaded successfully.")
