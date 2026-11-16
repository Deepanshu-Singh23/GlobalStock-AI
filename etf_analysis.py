
import pandas as pd
import numpy as np
import yfinance as yf


def safe_percentage(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value * 100
    except (TypeError, ValueError):
        pass
    return None


def get_etf_analysis(symbol):

    result = {
        "symbol": symbol,
        "top_holdings": pd.DataFrame(),
        "sector_weightings": {},
        "asset_classes": {},
        "bond_holdings": pd.DataFrame(),
        "status": {}
    }

    try:
        ticker = yf.Ticker(symbol)
        funds = ticker.funds_data

        # Top holdings
        try:
            holdings = funds.top_holdings

            if isinstance(holdings, pd.DataFrame) and not holdings.empty:
                holdings = holdings.copy()

                if "Holding Percent" in holdings.columns:
                    holdings["Weight (%)"] = (
                        pd.to_numeric(
                            holdings["Holding Percent"],
                            errors="coerce"
                        ) * 100
                    )

                holdings.index.name = "Symbol"
                result["top_holdings"] = holdings

            result["status"]["holdings"] = (
                "Available" if not result["top_holdings"].empty
                else "Not available from provider"
            )

        except Exception as e:
            result["status"]["holdings"] = str(e)

        # Sector weightings
        try:
            sectors = funds.sector_weightings

            if isinstance(sectors, dict):
                result["sector_weightings"] = {
                    key: safe_percentage(value)
                    for key, value in sectors.items()
                }

            result["status"]["sectors"] = (
                "Available" if result["sector_weightings"]
                else "Not available from provider"
            )

        except Exception as e:
            result["status"]["sectors"] = str(e)

        # Asset classes
        try:
            assets = funds.asset_classes

            if isinstance(assets, dict):
                result["asset_classes"] = {
                    key: safe_percentage(value)
                    for key, value in assets.items()
                }

            result["status"]["assets"] = (
                "Available" if result["asset_classes"]
                else "Not available from provider"
            )

        except Exception as e:
            result["status"]["assets"] = str(e)

        # Bond holdings
        try:
            bonds = funds.bond_holdings

            if isinstance(bonds, pd.DataFrame):
                result["bond_holdings"] = bonds.copy()

            result["status"]["bonds"] = (
                "Available" if not result["bond_holdings"].empty
                else "Partial or unavailable"
            )

        except Exception as e:
            result["status"]["bonds"] = str(e)

    except Exception as e:
        result["status"]["general"] = str(e)

    return result


def compare_etfs(symbols):

    rows = []

    for symbol in symbols:

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.get_info()

            rows.append({
                "Symbol": symbol,
                "Fund Name": (
                    info.get("longName")
                    or info.get("shortName")
                    or "Not available"
                ),
                "NAV": info.get("navPrice"),
                "Total Assets": (
                    info.get("totalAssets")
                    or info.get("netAssets")
                ),
                "Expense Ratio (%)": (
                    info.get("netExpenseRatio")
                ),
                "3-Month NAV Return (%)": (
                    info.get("trailingThreeMonthNavReturns")
                ),
                "Fund Family": info.get("fundFamily"),
                "Category": info.get("category")
            })

        except Exception as e:
            rows.append({
                "Symbol": symbol,
                "Fund Name": "Data unavailable",
                "Error": str(e)
            })

    return pd.DataFrame(rows)


def calculate_holdings_overlap(data_a, data_b):

    holdings_a = data_a.get("top_holdings", pd.DataFrame())
    holdings_b = data_b.get("top_holdings", pd.DataFrame())

    if holdings_a.empty or holdings_b.empty:
        return {
            "available": False,
            "message": (
                "Holdings overlap cannot be calculated because "
                "one or both ETF holdings lists are unavailable."
            ),
            "overlap": pd.DataFrame()
        }

    def prepare_weights(df):

        temp = df.copy()

        if "Holding Percent" in temp.columns:
            weights = pd.to_numeric(
                temp["Holding Percent"],
                errors="coerce"
            )

        elif "Weight (%)" in temp.columns:
            weights = pd.to_numeric(
                temp["Weight (%)"],
                errors="coerce"
            ) / 100

        else:
            return pd.Series(dtype=float)

        temp["weight"] = weights

        temp.index = temp.index.astype(str).str.upper()

        return temp["weight"].dropna()

    weights_a = prepare_weights(holdings_a)
    weights_b = prepare_weights(holdings_b)

    common = sorted(set(weights_a.index) & set(weights_b.index))

    if not common:
        return {
            "available": True,
            "message": "No common holdings found in the returned lists.",
            "overlap": pd.DataFrame()
        }

    overlap = pd.DataFrame({
        "ETF A Weight (%)": weights_a.loc[common] * 100,
        "ETF B Weight (%)": weights_b.loc[common] * 100
    })

    overlap["Lower Shared Weight (%)"] = (
        overlap[["ETF A Weight (%)", "ETF B Weight (%)"]].min(axis=1)
    )

    overlap.index.name = "Common Holding"

    return {
        "available": True,
        "message": "Overlap calculated from provider-returned holdings.",
        "overlap": overlap.sort_values(
            "Lower Shared Weight (%)",
            ascending=False
        )
    }


print("ETF Analysis module created successfully!")

