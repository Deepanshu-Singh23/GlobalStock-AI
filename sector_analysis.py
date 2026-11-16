import os
import glob
import numpy as np
import pandas as pd


def load_sector_data(sector_path):
    df = pd.read_csv(sector_path)
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["sector"] = df["sector"].fillna("Unknown").astype(str).str.strip()
    df.loc[df["sector"] == "", "sector"] = "Unknown"
    return df


def load_price_data(price_path, symbol):
    filename = symbol.replace(".", "_") + ".csv"
    filepath = os.path.join(price_path, filename)

    if not os.path.exists(filepath):
        return None

    try:
        df = pd.read_csv(filepath)

        if "date" not in df.columns or "adj_close" not in df.columns:
            return None

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")

        df = df.dropna(subset=["date", "adj_close"])
        df = df[df["adj_close"] > 0]
        df = df.sort_values("date").drop_duplicates("date")

        if len(df) < 20:
            return None

        df["daily_return"] = df["adj_close"].pct_change()
        df = df.dropna(subset=["daily_return"])

        return df

    except Exception:
        return None


def build_stock_metrics(sector_df, price_path):
    records = []

    for _, row in sector_df.iterrows():
        symbol = row["symbol"]
        prices = load_price_data(price_path, symbol)

        if prices is None or len(prices) < 19:
            continue

        returns = prices["daily_return"]
        equity_curve = (1 + returns).cumprod()
        drawdown = equity_curve / equity_curve.cummax() - 1

        first_price = prices["adj_close"].iloc[0]
        last_price = prices["adj_close"].iloc[-1]

        total_return = (last_price / first_price - 1) * 100
        annualized_volatility = returns.std() * np.sqrt(252) * 100
        max_drawdown = drawdown.min() * 100

        records.append({
            "symbol": symbol,
            "company": row.get("company", "Unknown"),
            "sector": row["sector"],
            "country": row.get("country", "Unknown"),
            "observations": len(prices),
            "start_date": prices["date"].min(),
            "end_date": prices["date"].max(),
            "total_return_pct": total_return,
            "annualized_volatility_pct": annualized_volatility,
            "max_drawdown_pct": max_drawdown
        })

    return pd.DataFrame(records)


def build_sector_summary(stock_metrics):
    if stock_metrics.empty:
        return pd.DataFrame()

    summary = (
        stock_metrics[stock_metrics["sector"] != "Unknown"]
        .groupby("sector")
        .agg(
            securities=("symbol", "count"),
            average_total_return_pct=("total_return_pct", "mean"),
            average_annualized_volatility_pct=(
                "annualized_volatility_pct", "mean"
            ),
            average_max_drawdown_pct=("max_drawdown_pct", "mean")
        )
        .reset_index()
        .sort_values("securities", ascending=False)
    )

    return summary


def build_sector_timeseries(sector_df, price_path):
    daily_frames = []

    for _, row in sector_df.iterrows():
        if row["sector"] == "Unknown":
            continue

        prices = load_price_data(price_path, row["symbol"])

        if prices is None:
            continue

        temp = prices[["date", "daily_return"]].copy()
        temp["sector"] = row["sector"]
        temp["symbol"] = row["symbol"]
        daily_frames.append(temp)

    if not daily_frames:
        return pd.DataFrame()

    all_returns = pd.concat(daily_frames, ignore_index=True)

    sector_daily = (
        all_returns.groupby(["sector", "date"])["daily_return"]
        .mean()
        .reset_index()
    )

    sector_daily["sector_index"] = (
        sector_daily.groupby("sector")["daily_return"]
        .transform(lambda x: (1 + x).cumprod() * 100)
    )

    return sector_daily