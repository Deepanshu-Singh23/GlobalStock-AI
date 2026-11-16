
import os
import pandas as pd
import numpy as np
import yfinance as yf

PROJECT_PATH = "/content/drive/MyDrive/GlobalStock_AI"
DATA_PATH = os.path.join(PROJECT_PATH, "data", "processed")
PORTFOLIO_PATH = os.path.join(DATA_PATH, "virtual_portfolio.csv")

os.makedirs(DATA_PATH, exist_ok=True)

def load_portfolio():
    if os.path.exists(PORTFOLIO_PATH):
        return pd.read_csv(PORTFOLIO_PATH)

    return pd.DataFrame(columns=["Symbol", "Quantity"])

def save_portfolio(data):
    data.to_csv(PORTFOLIO_PATH, index=False)

def add_stock(symbol, quantity):
    symbol = symbol.strip().upper()
    quantity = float(quantity)

    if not symbol or quantity <= 0:
        raise ValueError("Enter valid symbol and quantity")

    portfolio = load_portfolio()

    if symbol in portfolio["Symbol"].values:
        portfolio.loc[
            portfolio["Symbol"] == symbol, "Quantity"
        ] += quantity
    else:
        portfolio.loc[len(portfolio)] = [symbol, quantity]

    save_portfolio(portfolio)
    return portfolio

def remove_stock(symbol):
    portfolio = load_portfolio()
    portfolio = portfolio[portfolio["Symbol"] != symbol]
    save_portfolio(portfolio)
    return portfolio

def clear_portfolio():
    empty = pd.DataFrame(columns=["Symbol", "Quantity"])
    save_portfolio(empty)
    return empty

def get_portfolio_prices(period="1y"):
    portfolio = load_portfolio()

    if portfolio.empty:
        return pd.DataFrame()

    symbols = portfolio["Symbol"].dropna().tolist()

    if not symbols:
        return pd.DataFrame()

    data = yf.download(
        symbols,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=symbols[0])
    else:
        if "Close" not in data.columns:
            return pd.DataFrame()
        prices = data[["Close"]].copy()
        prices.columns = [symbols[0]]

    prices.columns = [str(c).upper() for c in prices.columns]

    return prices.apply(
        pd.to_numeric, errors="coerce"
    ).ffill().dropna(how="all")

def calculate_portfolio(prices):
    portfolio = load_portfolio()

    if portfolio.empty or prices.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    quantities = portfolio.set_index("Symbol")["Quantity"]
    quantities.index = quantities.index.str.upper()

    available = [
        s for s in quantities.index if s in prices.columns
    ]

    if not available:
        return pd.Series(dtype=float), pd.DataFrame()

    selected_prices = prices[available].copy()
    selected_prices = selected_prices.ffill().dropna(how="all")

    values = selected_prices.mul(
        quantities[available], axis=1
    )

    total_value = values.sum(axis=1, min_count=1).dropna()

    if total_value.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    latest = values.iloc[-1].dropna()
    total = latest.sum()

    allocation = pd.DataFrame({
        "Symbol": latest.index,
        "Portfolio Value": latest.values
    })

    allocation["Allocation (%)"] = (
        latest.values / total * 100 if total > 0 else 0
    )

    return total_value, allocation

def calculate_portfolio_metrics(values):
    if values.empty or len(values) < 2:
        return {}

    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) < 2 or values.iloc[0] == 0:
        return {}

    daily_returns = values.pct_change().replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    volatility = (
        daily_returns.std() * np.sqrt(252) * 100
        if len(daily_returns) > 1 else 0
    )

    running_max = values.cummax()
    drawdown = (values / running_max - 1) * 100

    total_return = (
        values.iloc[-1] / values.iloc[0] - 1
    ) * 100

    return {
        "Historical Change (%)": round(total_return, 2),
        "Annualized Volatility (%)": round(volatility, 2),
        "Maximum Drawdown (%)": round(drawdown.min(), 2)
    }
