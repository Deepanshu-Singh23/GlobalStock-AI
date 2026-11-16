
import os
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_PATH = "/content/drive/MyDrive/GlobalStock_AI"

REPORT_PATH = os.path.join(
    PROJECT_PATH, "data", "processed", "research_reports"
)

os.makedirs(REPORT_PATH, exist_ok=True)


def valid_number(value):
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return None


def format_amount(value):
    number = valid_number(value)

    if number is None:
        return "Not available"

    if abs(number) >= 1_000_000_000_000:
        return f"{number / 1_000_000_000_000:.2f} trillion"
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} billion"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f} million"

    return f"{number:,.2f}"


def detect_security_type(fundamentals):
    info = fundamentals or {}

    quote_type = str(info.get("quoteType", "")).upper()
    category = str(info.get("category", "")).lower()
    name = str(
        info.get("longName") or info.get("shortName") or ""
    ).lower()

    fund_types = {
        "ETF", "MUTUALFUND", "MONEYMARKET"
    }

    if quote_type in fund_types:
        return "Fund / ETF"

    if (
        info.get("fundFamily")
        or info.get("annualReportExpenseRatio") is not None
        or info.get("navPrice") is not None
        or "etf" in name
        or "fund" in name
        or "clo" in category
    ):
        return "Fund / ETF"

    if quote_type == "EQUITY":
        return "Company"

    if quote_type in {"INDEX", "FUTURE", "CURRENCY", "CRYPTOCURRENCY"}:
        return "Other Security"

    if info.get("totalRevenue") is not None or info.get("marketCap") is not None:
        return "Company"

    return "Other Security"


def calculate_growth(df, column):
    if (
        df is None
        or df.empty
        or column not in df.columns
        or "period" not in df.columns
    ):
        return None

    temp = df[["period", column]].copy()
    temp[column] = pd.to_numeric(temp[column], errors="coerce")
    temp = temp.dropna().sort_values("period")

    if len(temp) < 2:
        return None

    previous = valid_number(temp.iloc[-2][column])
    latest = valid_number(temp.iloc[-1][column])

    if previous is None or latest is None or previous == 0:
        return None

    return ((latest - previous) / abs(previous)) * 100


def availability_score(fundamentals, keys):
    available = 0

    for key in keys:
        aliases = key if isinstance(key, tuple) else (key,)

        if any(
            valid_number(fundamentals.get(alias)) is not None
            for alias in aliases
        ):
            available += 1

    applicable = len(keys)
    score = (available / applicable * 100) if applicable else 0

    return available, applicable, score


def build_research_report(
    symbol,
    fundamentals,
    annual_data,
    quarterly_data
):
    fundamentals = fundamentals or {}

    annual_data = (
        annual_data if isinstance(annual_data, pd.DataFrame)
        else pd.DataFrame()
    )

    quarterly_data = (
        quarterly_data if isinstance(quarterly_data, pd.DataFrame)
        else pd.DataFrame()
    )

    security_type = detect_security_type(fundamentals)

    name = (
        fundamentals.get("longName")
        or fundamentals.get("shortName")
        or symbol
    )

    sector = fundamentals.get("sector") or "Not available"
    industry = fundamentals.get("industry") or "Not available"

    report = [
        f"# Research Report: {name}",
        f"**Ticker:** {symbol}",
        f"**Security Type:** {security_type}",
        f"**Sector:** {sector}",
        f"**Industry:** {industry}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    ]

    report.append("\n## 1. Security Overview")

    description = (
        fundamentals.get("longBusinessSummary")
        or fundamentals.get("description")
        or "Description is not available."
    )

    report.append(str(description))

    if security_type == "Company":

        report.append("\n## 2. Company Valuation and Financial Metrics")

        metric_labels = {
            "marketCap": ("Market Capitalization", "amount"),
            "trailingPE": ("Trailing P/E", "number"),
            "forwardPE": ("Forward P/E", "number"),
            "priceToBook": ("Price to Book", "number"),
            "trailingEps": ("Trailing EPS", "number"),
            "totalRevenue": ("Reported Revenue", "amount"),
            "netIncomeToCommon": ("Net Income", "amount"),
            "profitMargins": ("Profit Margin", "percent"),
            "returnOnEquity": ("Return on Equity", "percent"),
            "beta": ("Beta", "number")
        }

        for key, (label, kind) in metric_labels.items():
            value = valid_number(fundamentals.get(key))

            if value is None:
                display = "Not available"
            elif kind == "amount":
                display = format_amount(value)
            elif kind == "percent":
                display = f"{value * 100:.2f}%"
            else:
                display = f"{value:.2f}"

            report.append(f"- **{label}:** {display}")

        score_keys = list(metric_labels.keys())

    elif security_type == "Fund / ETF":

        report.append("\n## 2. Fund Profile and Metrics")

        fund_metrics = [
            (
                "NAV",
                ("navPrice",),
                "number",
                "normal"
            ),
            (
                "Total Fund Assets",
                ("totalAssets", "netAssets"),
                "amount",
                "normal"
            ),
            (
                "Annual Expense Ratio",
                ("netExpenseRatio", "annualReportExpenseRatio"),
                "percent",
                "already_percent"
            ),
            (
                "Distribution Yield",
                ("trailingAnnualDividendYield", "dividendYield", "yield"),
                "percent",
                "fraction"
            ),
            (
                "3-Month NAV Return",
                ("trailingThreeMonthNavReturns", "trailingThreeMonthReturns"),
                "percent",
                "already_percent"
            ),
            (
                "3-Year Beta",
                ("beta3Year",),
                "number",
                "normal"
            ),
            (
                "52-Week High",
                ("fiftyTwoWeekHigh",),
                "number",
                "normal"
            ),
            (
                "52-Week Low",
                ("fiftyTwoWeekLow",),
                "number",
                "normal"
            )
        ]

        score_keys = []

        for label, aliases, kind, percent_mode in fund_metrics:

            value = None

            for key in aliases:
                candidate = valid_number(fundamentals.get(key))

                if candidate is not None:
                    value = candidate
                    break

            score_keys.append(aliases)

            if value is None:
                display = "Not available"

            elif kind == "amount":
                display = format_amount(value)

            elif kind == "percent":
                if percent_mode == "fraction":
                    display = f"{value * 100:.2f}%"
                else:
                    display = f"{value:.2f}%"

            else:
                display = f"{value:.4f}" if label == "NAV" else f"{value:.2f}"

            report.append(f"- **{label}:** {display}")

        fund_family = fundamentals.get("fundFamily")
        category = fundamentals.get("category")

        report.append(
            f"- **Fund Family:** {fund_family or 'Not available'}"
        )
        report.append(
            f"- **Fund Category:** {category or 'Not available'}"
        )

    else:

        report.append("\n## 2. Available Security Metrics")

        other_metrics = {
            "regularMarketPrice": "Latest Available Price",
            "marketCap": "Market Capitalization",
            "volume": "Trading Volume",
            "averageVolume": "Average Volume",
            "currency": "Currency"
        }

        for key, label in other_metrics.items():
            value = fundamentals.get(key)

            if key == "currency":
                display = value or "Not available"
            else:
                number = valid_number(value)
                display = (
                    format_amount(number)
                    if number is not None
                    else "Not available"
                )

            report.append(f"- **{label}:** {display}")

        score_keys = [
            "regularMarketPrice",
            "marketCap",
            "volume",
            "averageVolume"
        ]

    available, applicable, score = availability_score(
        fundamentals, score_keys
    )

    report.append("\n## 3. Data Availability")

    report.append(
        f"- **Available metrics:** {available}/{applicable}"
    )
    report.append(
        f"- **Data availability score:** {score:.1f}%"
    )

    if score >= 75:
        report.append("- **Coverage:** High")
    elif score >= 40:
        report.append("- **Coverage:** Moderate")
    else:
        report.append("- **Coverage:** Limited")

    report.append("\n## 4. Financial Performance Trends")

    if security_type == "Company":

        metrics = [
            ("Annual Revenue Change",
             calculate_growth(annual_data, "total_revenue")),
            ("Annual Net Income Change",
             calculate_growth(annual_data, "net_income")),
            ("Quarterly Revenue Change",
             calculate_growth(quarterly_data, "total_revenue")),
            ("Quarterly Net Income Change",
             calculate_growth(quarterly_data, "net_income"))
        ]

        for label, value in metrics:
            display = (
                f"{value:+.2f}%"
                if value is not None
                else "Insufficient comparable data"
            )
            report.append(f"- **{label}:** {display}")

    elif security_type == "Fund / ETF":

        report.append(
            "Traditional company revenue and net income analysis "
            "is not applicable to this fund profile."
        )
        report.append(
            "Fund performance and holdings analysis requires "
            "reliable fund-specific historical data."
        )

    else:

        report.append(
            "Standard company financial trends are not applicable "
            "to this security type."
        )

    report.append("\n## 5. Data-Based Observations")

    observations = []

    if security_type == "Company":
        revenue_growth = calculate_growth(
            annual_data, "total_revenue"
        )
        income_growth = calculate_growth(
            annual_data, "net_income"
        )

        if revenue_growth is not None:
            observations.append(
                f"Revenue changed by {revenue_growth:+.2f}% "
                "between the two latest available annual periods."
            )

        if income_growth is not None:
            observations.append(
                f"Net income changed by {income_growth:+.2f}% "
                "between the two latest available annual periods."
            )

    if security_type == "Fund / ETF":
        observations.append(
            "This security is classified as a fund. "
            "Company-specific valuation ratios should not be "
            "used as its primary assessment metrics."
        )

    if not observations:
        observations.append(
            "Available data is insufficient for reliable "
            "comparative observations."
        )

    for item in observations:
        report.append(f"- {item}")

    report.append("\n## 6. Data Limitations")
    report.append(
        "- Metrics depend on the information supplied by the data provider."
    )
    report.append(
        "- Missing values are shown as unavailable and are not estimated."
    )
    report.append(
        "- Data availability score measures field coverage, not data accuracy."
    )
    report.append(
        "- Period comparisons use the latest two available periods "
        "and are not necessarily year-over-year."
    )
    report.append(
        "- This report is educational research, not investment advice."
    )

    return "\n".join(report)


def save_research_report(symbol, report):
    safe_symbol = (
        symbol.replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    file_path = os.path.join(
        REPORT_PATH, safe_symbol + "_research.md"
    )

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(report)

    return file_path
