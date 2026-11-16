import pandas as pd
import numpy as np

from fundamentals import get_fundamentals
from financials import get_financial_statements


def safe_number(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def revenue_growth(annual):

    if annual is None or annual.empty:
        return None

    if "total_revenue" not in annual.columns:
        return None

    data = annual[["period", "total_revenue"]].copy()
    data["total_revenue"] = pd.to_numeric(
        data["total_revenue"], errors="coerce"
    )
    data = data.dropna().sort_values("period")

    if len(data) < 2:
        return None

    previous = safe_number(data.iloc[-2]["total_revenue"])
    latest = safe_number(data.iloc[-1]["total_revenue"])

    if previous is None or latest is None or previous == 0:
        return None

    return ((latest - previous) / abs(previous)) * 100


def compare_companies(symbols):

    results = []
    errors = []

    for symbol in symbols:

        try:
            info, info_status = get_fundamentals(symbol)

            annual, quarterly, financial_status = (
                get_financial_statements(symbol)
            )

            if not info:
                errors.append(f"{symbol}: {info_status}")
                continue

            results.append({
                "Symbol": symbol,
                "Company": (
                    info.get("longName")
                    or info.get("shortName")
                    or symbol
                ),
                "Currency": info.get("currency") or "N/A",
                "Market Cap": safe_number(info.get("marketCap")),
                "P/E Ratio": safe_number(info.get("trailingPE")),
                "EPS": safe_number(info.get("trailingEps")),
                "Revenue": safe_number(info.get("totalRevenue")),
                "Net Income": safe_number(
                    info.get("netIncomeToCommon")
                ),
                "Profit Margin (%)": (
                    safe_number(info.get("profitMargins")) * 100
                    if safe_number(info.get("profitMargins")) is not None
                    else None
                ),
                "Annual Revenue Change (%)": revenue_growth(annual)
            })

        except Exception as error:
            errors.append(f"{symbol}: {str(error)}")

    return pd.DataFrame(results), errors