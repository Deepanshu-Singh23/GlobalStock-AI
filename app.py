import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import os
import numpy as np



PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(PROJECT_PATH, "data", "processed")
PRICE_PATH = "/tmp/globalstock_ai/prices"

os.makedirs(PRICE_PATH, exist_ok=True)


MASTER_FILE = os.path.join(DATA_PATH, "global_stock_master.csv")
SCREENER_FILE = os.path.join(DATA_PATH, "technical_screener_results.csv")


st.set_page_config(
    page_title="GlobalStock AI",
    page_icon="📈",
    layout="wide"
)

st.title("🌍 GlobalStock AI")
st.caption("Global Stock Market Research & Analytics Platform")
st.info("Historical market data only. Prices may be delayed or unavailable.")

@st.cache_data(ttl=3600)
def load_master():
    return pd.read_csv(MASTER_FILE)

@st.cache_data(ttl=3600)
def load_screener():
    if os.path.exists(SCREENER_FILE):
        return pd.read_csv(SCREENER_FILE)
    return pd.DataFrame()

def get_yahoo_symbol(row):
    symbol = str(row["symbol"]).strip()
    country = str(row["country"]).strip().lower()
    exchange = str(row["exchange_code"]).strip().upper()

    if country == "india":
        if exchange == "NSE" and not symbol.endswith(".NS"):
            return symbol + ".NS"
        if exchange == "BSE" and not symbol.endswith(".BO"):
            return symbol + ".BO"

    if country in ["uk", "united kingdom"] and not symbol.endswith(".L"):
        return symbol + ".L"

    if country == "japan" and not symbol.endswith(".T"):
        return symbol + ".T"

    if country in ["hong kong", "hongkong"] and not symbol.endswith(".HK"):
        return symbol + ".HK"

    return symbol

def get_price_file(symbol):
    safe_name = symbol.replace(".", "_").replace("/", "_")
    return os.path.join(PRICE_PATH, safe_name + ".csv")

def calculate_indicators(df):
    df = df.copy()

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date")

    df["SMA_20"] = df["close"].rolling(20).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["SMA_200"] = df["close"].rolling(200).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df["daily_return"] = df["close"].pct_change()
    df["volatility_20"] = (
        df["daily_return"].rolling(20).std() * np.sqrt(252)
    )

    return df

def normalize_prices(df):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            values = [
                str(x).strip().lower()
                for x in df.columns.get_level_values(level)
            ]
            if "close" in values:
                df.columns = df.columns.get_level_values(level)
                break

    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    if "date" not in df.columns:
        df = df.reset_index()
        df.columns = [
            str(c).strip().lower().replace(" ", "_")
            for c in df.columns
        ]

    if "date" not in df.columns:
        raise ValueError("Date column missing in downloaded data.")

    if "close" not in df.columns:
        raise ValueError("Close price missing in downloaded data.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"])

    return df

@st.cache_data(ttl=3600)
def download_stock_data(symbol):
    data = yf.download(
        symbol,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if data.empty:
        return pd.DataFrame()

    data = normalize_prices(data)
    data = calculate_indicators(data)

    file_path = get_price_file(symbol)
    data.to_csv(file_path, index=False)

    return data

def load_stock_prices(symbol, force_refresh=False):
    file_path = get_price_file(symbol)

    if force_refresh:
        download_stock_data.clear()

    if os.path.exists(file_path) and not force_refresh:
        try:
            data = pd.read_csv(file_path)
            data = normalize_prices(data)
            data = calculate_indicators(data)
            return data, "Loaded from saved database"
        except Exception:
            pass

    data = download_stock_data(symbol)

    if data.empty:
        return data, "No historical data returned by Yahoo Finance"

    return data, "Fresh data downloaded and saved to Drive"

if not os.path.exists(MASTER_FILE):
    st.error("Stock database not found. Run database cells first.")
    st.stop()

stocks = load_master()

# Sidebar filters
st.sidebar.header("Market Filters")

countries = ["All"] + sorted(
    stocks["country"].dropna().astype(str).unique().tolist()
)

selected_country = st.sidebar.selectbox(
    "Select Country", countries
)

filtered = stocks.copy()

if selected_country != "All":
    filtered = filtered[
        filtered["country"].astype(str) == selected_country
    ]

exchanges = ["All"] + sorted(
    filtered["exchange_code"].dropna().astype(str).unique().tolist()
)

selected_exchange = st.sidebar.selectbox(
    "Select Exchange", exchanges
)

if selected_exchange != "All":
    filtered = filtered[
        filtered["exchange_code"].astype(str) == selected_exchange
    ]

# Overview
c1, c2, c3 = st.columns(3)

c1.metric("Stock Listings", f"{len(stocks):,}")
c2.metric("Countries", stocks["country"].nunique())
c3.metric("Exchanges", stocks["exchange_code"].nunique())

st.divider()

# Search
st.subheader("🔎 Search Global Stocks")

query = st.text_input(
    "Search by company name or ticker",
    placeholder="Example: Reliance, TCS, AAPL"
)

if query:
    mask = (
        filtered["symbol"].astype(str).str.contains(
            query, case=False, na=False
        ) |
        filtered["company_name"].astype(str).str.contains(
            query, case=False, na=False
        )
    )
    filtered = filtered[mask]

st.dataframe(
    filtered.head(200),
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"Showing {len(filtered)} matching listings "
    "(maximum 200 displayed)."
)

st.divider()

# Stock analysis
st.subheader("📊 Stock Analysis")

if not filtered.empty:

    options = filtered.to_dict("records")

    selected = st.selectbox(
        "Choose a stock",
        options,
        format_func=lambda x: (
            f"{x['symbol']} — {x['company_name']} "
            f"({x['exchange_code']})"
        )
    )

    yahoo_symbol = get_yahoo_symbol(selected)

    st.caption(f"Yahoo Finance symbol: {yahoo_symbol}")

    force_refresh = st.button("🔄 Refresh Historical Data")

    with st.spinner(
        f"Checking historical data for {yahoo_symbol}..."
    ):
        try:
            prices, data_status = load_stock_prices(
                yahoo_symbol,
                force_refresh=force_refresh
            )

            st.caption(data_status)

            if not prices.empty:

                latest = prices.iloc[-1]

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Latest Close",
                    f"{latest['close']:,.2f}"
                )

                rsi = latest.get("RSI_14", np.nan)

                m2.metric(
                    "RSI (14)",
                    f"{rsi:.2f}" if pd.notna(rsi) else "N/A"
                )

                daily = latest.get("daily_return", np.nan)

                m3.metric(
                    "Daily Return",
                    f"{daily * 100:.2f}%"
                    if pd.notna(daily) else "N/A"
                )

                # Historical price chart
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=prices["date"],
                    y=prices["close"],
                    name="Close Price",
                    mode="lines"
                ))

                for column in ["SMA_20", "SMA_50", "EMA_20"]:
                    fig.add_trace(go.Scatter(
                        x=prices["date"],
                        y=prices[column],
                        name=column,
                        mode="lines"
                    ))

                fig.update_layout(
                    title=f"{selected['company_name']} — Historical Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template="plotly_dark",
                    hovermode="x unified"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # RSI chart
                rsi_fig = go.Figure()

                rsi_fig.add_trace(go.Scatter(
                    x=prices["date"],
                    y=prices["RSI_14"],
                    name="RSI (14)",
                    mode="lines"
                ))

                rsi_fig.add_hline(
                    y=70, line_dash="dash"
                )

                rsi_fig.add_hline(
                    y=30, line_dash="dash"
                )

                rsi_fig.update_layout(
                    title="Relative Strength Index",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    rsi_fig,
                    use_container_width=True
                )

                st.subheader("Historical Data")
                st.dataframe(
                    prices.tail(20).sort_values(
                        "date", ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.warning(
                    f"No historical data available for {yahoo_symbol}. "
                    "Yahoo Finance may not support this listing."
                )

        except Exception as e:
            st.error(f"Data loading error: {str(e)}")

else:
    st.warning("No matching stocks found.")

st.divider()



# AI RESEARCH MODULES START

st.divider()
st.header("🏢 Company Fundamentals")

from fundamentals import get_fundamentals
from financials import get_financial_statements
from research_assistant import (
    build_research_report,
    save_research_report
)

if not filtered.empty:

    research_symbol = get_yahoo_symbol(selected)

    st.caption(f"Company: {selected['company_name']}")
    st.caption(f"Yahoo Finance Symbol: {research_symbol}")

    # Company Fundamentals
    with st.spinner("Loading company fundamentals..."):

        try:
            fundamentals_result = get_fundamentals(research_symbol)

            if isinstance(fundamentals_result, tuple):
                research_fundamentals = fundamentals_result[0]
                fundamentals_status = fundamentals_result[1]
            else:
                research_fundamentals = fundamentals_result
                fundamentals_status = "Fundamentals loaded"

            if not isinstance(research_fundamentals, dict):
                research_fundamentals = {}

            st.caption(str(fundamentals_status))

            if research_fundamentals:

                st.subheader("Company Overview")

                overview_fields = [
                    "longName", "shortName", "symbol",
                    "sector", "industry", "country",
                    "currency", "marketCap", "trailingPE",
                    "forwardPE", "trailingEps", "totalRevenue",
                    "netIncomeToCommon", "profitMargins",
                    "operatingMargins", "beta",
                    "fiftyTwoWeekHigh", "fiftyTwoWeekLow"
                ]

                available_fields = {
                    key: research_fundamentals[key]
                    for key in overview_fields
                    if research_fundamentals.get(key) is not None
                }

                if available_fields:
                    fundamentals_df = pd.DataFrame(
                        available_fields.items(),
                        columns=["Metric", "Value"]
                    )

                    st.dataframe(
                        fundamentals_df,
                        use_container_width=True,
                        hide_index=True
                    )

                summary = research_fundamentals.get(
                    "longBusinessSummary"
                )

                if summary:
                    with st.expander("Company Business Summary"):
                        st.write(summary)

            else:
                st.warning("Fundamental data is unavailable.")

        except Exception as error:
            research_fundamentals = {}
            st.error(f"Fundamentals error: {error}")

    # Financial Performance
    st.divider()
    st.header("📈 Financial Performance Analysis")

    try:
        with st.spinner("Loading financial statements..."):

            financial_result = get_financial_statements(
                research_symbol
            )

            if isinstance(financial_result, tuple):
                annual_data = financial_result[0]
                quarterly_data = financial_result[1]
                financial_status = (
                    financial_result[2]
                    if len(financial_result) > 2
                    else "Financial data loaded"
                )
            else:
                annual_data = pd.DataFrame()
                quarterly_data = pd.DataFrame()
                financial_status = "Unexpected financial data format"

        st.caption(str(financial_status))

        if isinstance(annual_data, pd.DataFrame) and not annual_data.empty:

            st.subheader("Annual Financial Statements")
            st.dataframe(
                annual_data,
                use_container_width=True
            )

            annual_chart_columns = [
                c for c in [
                    "Total Revenue",
                    "Net Income",
                    "Operating Income",
                    "Gross Profit"
                ]
                if c in annual_data.columns
            ]

            if annual_chart_columns:
                st.subheader("Annual Financial Trends")
                st.line_chart(
                    annual_data[annual_chart_columns]
                )

        else:
            st.info("Annual financial statement data unavailable.")

        if isinstance(quarterly_data, pd.DataFrame) and not quarterly_data.empty:

            st.subheader("Quarterly Financial Statements")
            st.dataframe(
                quarterly_data,
                use_container_width=True
            )

        else:
            st.info("Quarterly financial statement data unavailable.")

    except Exception as error:
        annual_data = pd.DataFrame()
        quarterly_data = pd.DataFrame()
        st.error(f"Financial analysis error: {error}")

    # AI Research Assistant
    st.divider()
    st.header("🤖 AI Research Assistant")

    st.caption(
        "Rule-based, data-driven educational company research. "
        "This is not an LLM and does not provide investment recommendations."
    )

    if st.button(
        "🧠 Generate Company Research Report",
        key="generate_company_research_report"
    ):

        try:
            with st.spinner("Analysing available company data..."):

                research_report = build_research_report(
                    research_symbol,
                    research_fundamentals,
                    annual_data,
                    quarterly_data
                )

                report_path = save_research_report(
                    research_symbol,
                    research_report
                )

            st.success("Research report generated successfully!")

            st.markdown(research_report)

            st.download_button(
                label="📥 Download Research Report",
                data=research_report,
                file_name=f"{research_symbol}_research.md",
                mime="text/markdown",
                key="download_company_research_report"
            )

            st.caption(f"Report saved to Drive: {report_path}")

        except Exception as error:
            st.error(f"Research report error: {error}")

else:
    st.warning("Select a valid stock to access company research.")

# AI RESEARCH MODULES END








# GLOBAL_MARKET_INTELLIGENCE_START

import os
import glob
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.divider()
st.header("🌍 Global Market Intelligence")

st.caption(
    "Global market analytics using available historical price data."
)

# ------------------------------------------------
# FIND PROJECT DATA DIRECTORY
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

possible_paths = [
    os.path.join(BASE_DIR, "data", "processed"),
    os.path.join(BASE_DIR, "data"),
    BASE_DIR
]

DATA_FOLDER = None

for folder in possible_paths:
    candidate = os.path.join(
        folder, "synchronized_market_data.csv"
    )

    if os.path.isfile(candidate):
        DATA_FOLDER = folder
        break

# Fallback: search project subdirectories
if DATA_FOLDER is None:
    matches = glob.glob(
        os.path.join(
            BASE_DIR,
            "**",
            "synchronized_market_data.csv"
        ),
        recursive=True
    )

    if matches:
        DATA_FOLDER = os.path.dirname(matches[0])

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

if DATA_FOLDER is None:

    st.error(
        "Synchronized market dataset not found."
    )

    st.info(
        "Check that synchronized_market_data.csv "
        "is uploaded inside data/processed."
    )

else:

    st.success(
        f"Market dataset located: {DATA_FOLDER}"
    )

    def load_csv(filename):

        path = os.path.join(DATA_FOLDER, filename)

        if os.path.isfile(path):
            try:
                return pd.read_csv(path)
            except Exception as error:
                st.warning(
                    f"Could not read {filename}: {error}"
                )

        return pd.DataFrame()

    market_df = load_csv(
        "synchronized_market_data.csv"
    )

    country_df = load_csv(
        "synchronized_country_analytics.csv"
    )

    exchange_df = load_csv(
        "synchronized_exchange_analytics.csv"
    )

    sector_df = load_csv(
        "sector_summary.csv"
    )

    sector_detail_df = load_csv(
        "sector_analysis.csv"
    )

    screener_df = load_csv(
        "technical_screener_results.csv"
    )

    # ------------------------------------------------
    # VALIDATE DATA
    # ------------------------------------------------

    required_columns = [
        "symbol",
        "country",
        "exchange_code",
        "latest_close",
        "daily_return_pct",
        "return_1m_pct",
        "return_3m_pct"
    ]

    if market_df.empty:

        st.warning(
            "Market dataset exists but contains no records."
        )

    elif not all(
        col in market_df.columns
        for col in required_columns
    ):

        st.error(
            "Required columns are missing from market dataset."
        )

        st.write(
            "Available columns:",
            market_df.columns.tolist()
        )

    else:

        # Convert numerical columns
        numeric_columns = [
            "latest_close",
            "daily_return_pct",
            "return_1m_pct",
            "return_3m_pct",
            "return_1y_pct"
        ]

        for col in numeric_columns:

            if col in market_df.columns:
                market_df[col] = pd.to_numeric(
                    market_df[col],
                    errors="coerce"
                )

        market_df = market_df.drop_duplicates(
            subset=["country", "symbol"]
        ).copy()

        if "price_date" in market_df.columns:

            market_df["price_date"] = pd.to_datetime(
                market_df["price_date"],
                errors="coerce"
            )

        # ------------------------------------------------
        # GLOBAL MARKET OVERVIEW
        # ------------------------------------------------

        st.subheader("1. Global Market Overview")

        total_stocks = len(market_df)

        countries = market_df["country"].nunique()

        exchanges = market_df["exchange_code"].nunique()

        advancing = (
            market_df["daily_return_pct"] > 0
        ).sum()

        declining = (
            market_df["daily_return_pct"] < 0
        ).sum()

        unchanged = (
            market_df["daily_return_pct"] == 0
        ).sum()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Securities",
            f"{total_stocks:,}"
        )

        c2.metric(
            "Countries",
            countries
        )

        c3.metric(
            "Exchanges",
            exchanges
        )

        c4.metric(
            "Advancing",
            f"{advancing:,}"
        )

        c5, c6, c7 = st.columns(3)

        c5.metric("Declining", f"{declining:,}")

        c6.metric("Unchanged", f"{unchanged:,}")

        c7.metric(
            "Declining Share",
            f"{declining / max(total_stocks, 1) * 100:.2f}%"
        )

        movement_fig = go.Figure(
            data=[
                go.Bar(
                    x=[
                        "Advancing",
                        "Declining",
                        "Unchanged"
                    ],
                    y=[
                        advancing,
                        declining,
                        unchanged
                    ]
                )
            ]
        )

        movement_fig.update_layout(
            title="Global Market Breadth",
            height=350
        )

        st.plotly_chart(
            movement_fig,
            use_container_width=True
        )

        # ------------------------------------------------
        # COUNTRY ANALYTICS
        # ------------------------------------------------

        st.divider()

        st.subheader("2. Country Analytics")

        if not country_df.empty:

            st.dataframe(
                country_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            country_summary = (
                market_df.groupby("country")
                .agg(
                    stocks_with_prices=("symbol", "nunique"),
                    median_daily_return=(
                        "daily_return_pct", "median"
                    )
                )
                .reset_index()
            )

            st.dataframe(
                country_summary,
                use_container_width=True,
                hide_index=True
            )

        # ------------------------------------------------
        # EXCHANGE ANALYTICS
        # ------------------------------------------------

        st.divider()

        st.subheader("3. Exchange Analytics")

        if not exchange_df.empty:

            st.dataframe(
                exchange_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            exchange_summary = (
                market_df.groupby("exchange_code")
                .agg(
                    stocks_with_prices=("symbol", "nunique"),
                    median_daily_return=(
                        "daily_return_pct", "median"
                    )
                )
                .reset_index()
            )

            st.dataframe(
                exchange_summary,
                use_container_width=True,
                hide_index=True
            )

        # ------------------------------------------------
        # SECTOR ANALYSIS
        # ------------------------------------------------

        st.divider()

        st.subheader("4. Sector Analysis")

        if not sector_df.empty:

            st.dataframe(
                sector_df,
                use_container_width=True,
                hide_index=True
            )

        elif "sector" in market_df.columns:

            sector_summary = (
                market_df.groupby("sector")
                .agg(
                    securities=("symbol", "nunique"),
                    median_return=(
                        "daily_return_pct", "median"
                    )
                )
                .reset_index()
            )

            st.dataframe(
                sector_summary,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Sector classification is unavailable."
            )

        if not sector_detail_df.empty:

            with st.expander(
                "View Sector Classification Details"
            ):

                st.dataframe(
                    sector_detail_df,
                    use_container_width=True
                )

        # ------------------------------------------------
        # GLOBAL TECHNICAL SCREENER
        # ------------------------------------------------

        st.divider()

        st.subheader("5. Global Technical Screener")

        if not screener_df.empty:

            if "symbol" in screener_df.columns:

                search = st.text_input(
                    "Search stock symbol",
                    key="global_market_search"
                )

                if search.strip():

                    screener_df = screener_df[
                        screener_df["symbol"]
                        .astype(str)
                        .str.contains(
                            search.strip(),
                            case=False,
                            na=False
                        )
                    ]

            st.dataframe(
                screener_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Technical screener data is unavailable."
            )

        # ------------------------------------------------
        # DOWNLOAD DATA
        # ------------------------------------------------

        st.divider()

        st.subheader("Download Market Dataset")

        st.download_button(
            "Download Global Market Data",
            data=market_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="global_market_data.csv",
            mime="text/csv"
        )

        st.success(
            "Global Market Intelligence loaded successfully."
        )

# GLOBAL_MARKET_INTELLIGENCE_END




# GLOBAL_MARKET_MOVERS_START

st.divider()
st.header("🚀 Global Market Movers")

st.caption(
    "Top market performers based on synchronized historical "
    "price data. Educational analytics only."
)

MOVERS_FILE = os.path.join(
    DATA_PATH,
    "synchronized_market_data.csv"
)

if os.path.exists(MOVERS_FILE):

    movers_df = pd.read_csv(MOVERS_FILE)

    required_columns = [
        "symbol", "company_name", "country",
        "exchange_code", "latest_close",
        "daily_return_pct", "return_1m_pct",
        "return_3m_pct"
    ]

    if all(col in movers_df.columns for col in required_columns):

        for col in [
            "latest_close", "daily_return_pct",
            "return_1m_pct", "return_3m_pct"
        ]:
            movers_df[col] = pd.to_numeric(
                movers_df[col], errors="coerce"
            )

        movers_df = movers_df.drop_duplicates(
            subset=["country", "symbol"]
        )

        movers_df["company_name"] = (
            movers_df["company_name"].fillna("N/A")
        )

        market_date = (
            movers_df["price_date"].max()
            if "price_date" in movers_df.columns
            else "Unavailable"
        )

        st.caption(f"Dataset market date: {market_date}")

        def show_movers_table(data, title):
            st.subheader(title)

            if data.empty:
                st.info("No sufficient data available.")
            else:
                display_columns = [
                    "symbol", "company_name", "country",
                    "exchange_code", "latest_close"
                ]

                return_col = [
                    c for c in data.columns
                    if c in [
                        "daily_return_pct",
                        "return_1m_pct",
                        "return_3m_pct"
                    ]
                ]

                display_columns += return_col

                st.dataframe(
                    data[display_columns].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True
                )

        tab1, tab2, tab3 = st.tabs([
            "Daily Movers",
            "1-Month Performers",
            "3-Month Performers"
        ])

        with tab1:
            daily_data = movers_df.dropna(
                subset=["daily_return_pct"]
            )

            col1, col2 = st.columns(2)

            with col1:
                show_movers_table(
                    daily_data.nlargest(
                        10, "daily_return_pct"
                    ),
                    "🏆 Top 10 Gainers"
                )

            with col2:
                show_movers_table(
                    daily_data.nsmallest(
                        10, "daily_return_pct"
                    ),
                    "Top 10 Decliners"
                )

        with tab2:
            monthly_data = movers_df.dropna(
                subset=["return_1m_pct"]
            )

            col1, col2 = st.columns(2)

            with col1:
                show_movers_table(
                    monthly_data.nlargest(
                        10, "return_1m_pct"
                    ),
                    "Top 10 One-Month Performers"
                )

            with col2:
                show_movers_table(
                    monthly_data.nsmallest(
                        10, "return_1m_pct"
                    ),
                    "Bottom 10 One-Month Performers"
                )

        with tab3:
            quarterly_data = movers_df.dropna(
                subset=["return_3m_pct"]
            )

            col1, col2 = st.columns(2)

            with col1:
                show_movers_table(
                    quarterly_data.nlargest(
                        10, "return_3m_pct"
                    ),
                    "Top 10 Three-Month Performers"
                )

            with col2:
                show_movers_table(
                    quarterly_data.nsmallest(
                        10, "return_3m_pct"
                    ),
                    "Bottom 10 Three-Month Performers"
                )

        st.divider()
        st.subheader("🌐 Country Market Breadth")

        breadth = movers_df.groupby("country").agg(
            securities=("symbol", "count"),
            advancing=(
                "daily_return_pct",
                lambda x: (x > 0).sum()
            ),
            declining=(
                "daily_return_pct",
                lambda x: (x < 0).sum()
            ),
            unchanged=(
                "daily_return_pct",
                lambda x: (x == 0).sum()
            )
        ).reset_index()

        st.dataframe(
            breadth,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning(
            "Required columns are missing from market dataset."
        )

else:
    st.warning(
        "Synchronized market dataset not found."
    )

# GLOBAL_MARKET_MOVERS_END



# GLOBAL_STOCK_DISCOVERY_START

st.divider()
st.header("🔎 Global Stock Discovery & Advanced Screener")

st.caption(
    "Explore available securities using market and historical "
    "performance filters. Educational analytics only."
)

discovery_file = os.path.join(
    DATA_PATH, "synchronized_market_data.csv"
)

if os.path.exists(discovery_file):
    df_discovery = pd.read_csv(discovery_file)

    required = [
        "symbol", "company_name", "country",
        "exchange_code", "latest_close",
        "daily_return_pct", "return_1m_pct",
        "return_3m_pct"
    ]

    if all(c in df_discovery.columns for c in required):

        for c in [
            "latest_close", "daily_return_pct",
            "return_1m_pct", "return_3m_pct"
        ]:
            df_discovery[c] = pd.to_numeric(
                df_discovery[c], errors="coerce"
            )

        df_discovery = df_discovery.drop_duplicates(
            subset=["country", "symbol"]
        ).copy()

        if "sector" not in df_discovery.columns:
            df_discovery["sector"] = "Unclassified"
        else:
            df_discovery["sector"] = (
                df_discovery["sector"].fillna("Unclassified")
            )

        if "asset_type" not in df_discovery.columns:
            df_discovery["asset_type"] = "Unknown"

        f1, f2, f3 = st.columns(3)

        with f1:
            country = st.selectbox(
                "Country",
                ["All"] + sorted(df_discovery["country"].dropna().unique()),
                key="discovery_country"
            )

        with f2:
            exchange = st.selectbox(
                "Exchange",
                ["All"] + sorted(df_discovery["exchange_code"].dropna().unique()),
                key="discovery_exchange"
            )

        with f3:
            sector = st.selectbox(
                "Sector",
                ["All"] + sorted(df_discovery["sector"].dropna().unique()),
                key="discovery_sector"
            )

        f4, f5 = st.columns(2)

        with f4:
            asset = st.selectbox(
                "Asset Type",
                ["All"] + sorted(df_discovery["asset_type"].dropna().unique()),
                key="discovery_asset"
            )

        with f5:
            query = st.text_input(
                "Search symbol or company",
                key="discovery_query"
            )

        metric_options = {
            "Daily Return (%)": "daily_return_pct",
            "1-Month Return (%)": "return_1m_pct",
            "3-Month Return (%)": "return_3m_pct"
        }

        metric_label = st.selectbox(
            "Performance Metric",
            list(metric_options.keys()),
            key="discovery_metric"
        )

        metric_col = metric_options[metric_label]

        result = df_discovery.copy()

        if country != "All":
            result = result[result["country"] == country]

        if exchange != "All":
            result = result[result["exchange_code"] == exchange]

        if sector != "All":
            result = result[result["sector"] == sector]

        if asset != "All":
            result = result[result["asset_type"] == asset]

        if query.strip():
            q = query.strip()
            mask = (
                result["symbol"].astype(str).str.contains(
                    q, case=False, na=False
                )
                |
                result["company_name"].astype(str).str.contains(
                    q, case=False, na=False
                )
            )
            result = result[mask]

        valid = result[metric_col].dropna()

        if not valid.empty:
            low = float(valid.min())
            high = float(valid.max())

            if low == high:
                low -= 0.5
                high += 0.5

            selected_range = st.slider(
                "Performance Range (%)",
                min_value=low,
                max_value=high,
                value=(low, high),
                key="discovery_range"
            )

            result = result[
                result[metric_col].between(
                    selected_range[0],
                    selected_range[1]
                )
            ]

        sort_choice = st.selectbox(
            "Sort Results",
            [
                "Highest Performance",
                "Lowest Performance",
                "Symbol A-Z"
            ],
            key="discovery_sort"
        )

        if sort_choice == "Highest Performance":
            result = result.sort_values(
                metric_col, ascending=False, na_position="last"
            )
        elif sort_choice == "Lowest Performance":
            result = result.sort_values(
                metric_col, ascending=True, na_position="last"
            )
        else:
            result = result.sort_values("symbol")

        st.subheader("Screening Results")
        st.metric("Matching Securities", len(result))

        columns = [
            "symbol", "company_name", "country",
            "exchange_code", "sector", "asset_type",
            "latest_close", "daily_return_pct",
            "return_1m_pct", "return_3m_pct"
        ]

        columns = [c for c in columns if c in result.columns]

        st.dataframe(
            result[columns],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Screening Results (CSV)",
            data=result[columns].to_csv(index=False).encode("utf-8"),
            file_name="global_stock_screening_results.csv",
            mime="text/csv"
        )

    else:
        st.warning("Required dataset columns are missing.")

else:
    st.warning("Synchronized market dataset not found.")

# GLOBAL_STOCK_DISCOVERY_END



# GLOBAL_MARKET_HEATMAP_START

st.divider()
st.header("🌡️ Global Market Heatmap")

st.caption(
    "Country and exchange-wise market performance based on synchronized "
    "historical price data. Green indicates positive returns; red indicates negative returns."
)

HEATMAP_FILE = os.path.join(DATA_PATH, "synchronized_market_data.csv")

if os.path.exists(HEATMAP_FILE):

    heatmap_df = pd.read_csv(HEATMAP_FILE)

    required_cols = [
        "country", "exchange_code", "symbol",
        "daily_return_pct", "return_1m_pct",
        "return_3m_pct", "return_1y_pct"
    ]

    if all(col in heatmap_df.columns for col in required_cols):

        metric_options = {
            "Daily Return": "daily_return_pct",
            "1-Month Return": "return_1m_pct",
            "3-Month Return": "return_3m_pct",
            "1-Year Return": "return_1y_pct"
        }

        selected_metric = st.selectbox(
            "Select performance period",
            list(metric_options.keys()),
            key="global_heatmap_metric"
        )

        metric_col = metric_options[selected_metric]

        heatmap_df[metric_col] = pd.to_numeric(
            heatmap_df[metric_col], errors="coerce"
        )

        country_exchange = (
            heatmap_df
            .dropna(subset=[metric_col])
            .groupby(["country", "exchange_code"])[metric_col]
            .median()
            .unstack()
        )

        if not country_exchange.empty:

            fig = go.Figure(
                data=go.Heatmap(
                    z=country_exchange.values,
                    x=country_exchange.columns.tolist(),
                    y=country_exchange.index.tolist(),
                    colorscale="RdYlGn",
                    zmid=0,
                    colorbar=dict(title="Median Return (%)"),
                    hovertemplate=(
                        "Country: %{y}<br>"
                        "Exchange: %{x}<br>"
                        "Median Return: %{z:.2f}%"
                        "<extra></extra>"
                    )
                )
            )

            fig.update_layout(
                title=f"{selected_metric} — Country vs Exchange",
                xaxis_title="Exchange",
                yaxis_title="Country",
                height=420
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Market Performance by Country & Exchange")

            summary = (
                heatmap_df
                .groupby(["country", "exchange_code"])
                .agg(
                    securities=("symbol", "nunique"),
                    median_return=(metric_col, "median")
                )
                .reset_index()
            )

            summary["median_return"] = summary["median_return"].round(2)

            st.dataframe(
                summary.sort_values("median_return", ascending=False),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Top Market Movers")

            countries = ["All Countries"] + sorted(
                heatmap_df["country"].dropna().unique().tolist()
            )

            selected_country = st.selectbox(
                "Filter country",
                countries,
                key="heatmap_country_filter"
            )

            movers = heatmap_df.dropna(subset=[metric_col]).copy()

            if selected_country != "All Countries":
                movers = movers[
                    movers["country"] == selected_country
                ]

            left, right = st.columns(2)

            display_cols = [
                "symbol", "company_name", "country",
                "exchange_code", metric_col
            ]

            display_cols = [
                col for col in display_cols if col in movers.columns
            ]

            with left:
                st.markdown("**Top 10 Gainers**")
                gainers = movers.nlargest(10, metric_col)
                st.dataframe(
                    gainers[display_cols],
                    use_container_width=True,
                    hide_index=True
                )

            with right:
                st.markdown("**Top 10 Decliners**")
                decliners = movers.nsmallest(10, metric_col)
                st.dataframe(
                    decliners[display_cols],
                    use_container_width=True,
                    hide_index=True
                )

        else:
            st.info("No sufficient return data available for this period.")

    else:
        st.warning("Required market data columns are missing.")

else:
    st.warning("Synchronized market dataset not found.")

# GLOBAL_MARKET_HEATMAP_END



# GLOBAL_MACRO_DASHBOARD_START

st.divider()
st.header("🌐 Global Macro & Economic Indicators")

st.caption(
    "Market-based macro indicators from Yahoo Finance. "
    "These are market proxies, not official GDP or inflation statistics."
)

@st.cache_data(ttl=1800, show_spinner=False)
def load_macro_indicators():

    instruments = {
        "US 10-Year Treasury Yield": "^TNX",
        "US 5-Year Treasury Yield": "^FVX",
        "US 13-Week Treasury Yield": "^IRX",
        "Market Volatility Index": "^VIX",
        "US Dollar Index": "DX-Y.NYB",
        "Gold Futures": "GC=F",
        "Crude Oil Futures": "CL=F"
    }

    records = []

    for name, symbol in instruments.items():
        try:
            history = yf.Ticker(symbol).history(
                period="6mo",
                interval="1d",
                auto_adjust=False
            )

            if history.empty or "Close" not in history.columns:
                continue

            close = pd.to_numeric(
                history["Close"], errors="coerce"
            ).dropna()

            if close.empty:
                continue

            latest = float(close.iloc[-1])

            one_month = (
                (latest / float(close.iloc[-22]) - 1) * 100
                if len(close) >= 22 and close.iloc[-22] != 0
                else np.nan
            )

            three_month = (
                (latest / float(close.iloc[-64]) - 1) * 100
                if len(close) >= 64 and close.iloc[-64] != 0
                else np.nan
            )

            records.append({
                "Indicator": name,
                "Symbol": symbol,
                "Latest Value": latest,
                "1M Change (%)": one_month,
                "3M Change (%)": three_month,
                "Data Date": close.index[-1].strftime("%Y-%m-%d")
            })

        except Exception:
            continue

    return pd.DataFrame(records)


if st.checkbox(
    "Load Global Macro Indicators",
    key="load_global_macro_indicators"
):

    with st.spinner("Loading macro market data..."):
        macro_df = load_macro_indicators()

    if not macro_df.empty:

        c1, c2, c3 = st.columns(3)

        for i, row in macro_df.head(3).iterrows():
            with [c1, c2, c3][i]:
                c1_metric = row["Latest Value"]
                st.metric(
                    row["Indicator"],
                    f"{c1_metric:,.2f}",
                    f"{row['1M Change (%)']:.2f}%"
                    if pd.notna(row["1M Change (%)"]) else None
                )

        st.subheader("Macro Indicator Overview")

        st.dataframe(
            macro_df.round(3),
            use_container_width=True,
            hide_index=True
        )

        chart_df = macro_df.dropna(subset=["1M Change (%)"])

        if not chart_df.empty:
            fig = go.Figure(
                go.Bar(
                    x=chart_df["Indicator"],
                    y=chart_df["1M Change (%)"],
                    text=chart_df["1M Change (%)"].round(2),
                    textposition="outside"
                )
            )

            fig.update_layout(
                title="One-Month Change in Macro Market Indicators (%)",
                xaxis_title="Indicator",
                yaxis_title="Change (%)",
                height=430
            )

            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Source: Yahoo Finance. Values may be delayed, unavailable "
            "or subject to provider limitations."
        )

    else:
        st.warning(
            "Macro data is currently unavailable. "
            "Try again later."
        )

# GLOBAL_MACRO_DASHBOARD_END



# ADVANCED_STOCK_SCREENER_V2_START

st.divider()
st.header("🔎 Advanced Stock Screener 2.0")

st.caption(
    "Filter available securities using historical returns, country, "
    "exchange and technical indicators."
)

market_file = os.path.join(DATA_PATH, "synchronized_market_data.csv")
technical_file = os.path.join(DATA_PATH, "technical_screener_results.csv")

screen_df = pd.DataFrame()

if os.path.exists(market_file):
    screen_df = pd.read_csv(market_file)

if not screen_df.empty:

    if os.path.exists(technical_file):
        technical_df = pd.read_csv(technical_file)

        if "symbol" in technical_df.columns:
            technical_df = technical_df.drop_duplicates(
                subset=["symbol"], keep="last"
            )

            extra_cols = [
                col for col in technical_df.columns
                if col not in screen_df.columns and col != "symbol"
            ]

            if extra_cols:
                screen_df = screen_df.merge(
                    technical_df[["symbol"] + extra_cols],
                    on="symbol",
                    how="left"
                )

    st.subheader("Screener Filters")

    f1, f2, f3 = st.columns(3)

    with f1:
        countries = sorted(
            screen_df["country"].dropna().astype(str).unique().tolist()
        )
        selected_countries = st.multiselect(
            "Country",
            countries,
            default=countries,
            key="advanced_screen_countries"
        )

    with f2:
        exchanges = sorted(
            screen_df["exchange_code"].dropna().astype(str).unique().tolist()
        )
        selected_exchanges = st.multiselect(
            "Exchange",
            exchanges,
            default=exchanges,
            key="advanced_screen_exchanges"
        )

    with f3:
        search_text = st.text_input(
            "Search symbol / company",
            key="advanced_screen_search"
        )

    return_options = {
        "Daily Return (%)": "daily_return_pct",
        "1-Month Return (%)": "return_1m_pct",
        "3-Month Return (%)": "return_3m_pct",
        "1-Year Return (%)": "return_1y_pct"
    }

    available_returns = {
        label: col for label, col in return_options.items()
        if col in screen_df.columns
    }

    r1, r2, r3 = st.columns(3)

    with r1:
        selected_return = st.selectbox(
            "Return metric",
            list(available_returns.keys()),
            key="advanced_screen_return_metric"
        )

    return_col = available_returns[selected_return]

    numeric_values = pd.to_numeric(
        screen_df[return_col], errors="coerce"
    ).dropna()

    if not numeric_values.empty:
        min_value = float(numeric_values.min())
        max_value = float(numeric_values.max())

        with r2:
            min_return = st.number_input(
                "Minimum return (%)",
                value=float(min_value),
                key="advanced_screen_min_return"
            )

        with r3:
            max_return = st.number_input(
                "Maximum return (%)",
                value=float(max_value),
                key="advanced_screen_max_return"
            )
    else:
        min_return, max_return = -100.0, 100.0

    filtered_screen = screen_df.copy()

    filtered_screen = filtered_screen[
        filtered_screen["country"].isin(selected_countries)
        & filtered_screen["exchange_code"].isin(selected_exchanges)
    ]

    if search_text.strip():
        search_mask = (
            filtered_screen["symbol"].astype(str).str.contains(
                search_text.strip(), case=False, na=False
            )
        )

        if "company_name" in filtered_screen.columns:
            search_mask |= filtered_screen["company_name"].astype(str).str.contains(
                search_text.strip(), case=False, na=False
            )

        filtered_screen = filtered_screen[search_mask]

    filtered_screen[return_col] = pd.to_numeric(
        filtered_screen[return_col], errors="coerce"
    )

    if min_return <= max_return:
        filtered_screen = filtered_screen[
            filtered_screen[return_col].between(
                min_return, max_return, inclusive="both"
            )
        ]
    else:
        st.warning("Minimum return cannot exceed maximum return.")

    st.subheader("Screening Results")

    m1, m2, m3 = st.columns(3)

    m1.metric("Matching Securities", f"{len(filtered_screen):,}")

    if not filtered_screen.empty:
        m2.metric(
            "Median Selected Return",
            f"{filtered_screen[return_col].median():.2f}%"
        )

        m3.metric(
            "Countries",
            filtered_screen["country"].nunique()
        )

        sort_choice = st.radio(
            "Sort results",
            ["Highest return first", "Lowest return first"],
            horizontal=True,
            key="advanced_screen_sort"
        )

        filtered_screen = filtered_screen.sort_values(
            return_col,
            ascending=(sort_choice == "Lowest return first")
        )

        st.dataframe(
            filtered_screen,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download Screened Results",
            data=filtered_screen.to_csv(index=False).encode("utf-8"),
            file_name="global_stock_screening_results.csv",
            mime="text/csv"
        )

    else:
        st.info("No securities match the selected filters.")

else:
    st.warning("Synchronized market data is unavailable.")

# ADVANCED_STOCK_SCREENER_V2_END



# PORTFOLIO_RISK_ANALYTICS_START

st.divider()
st.header("🛡️ Portfolio Risk Analytics")

st.caption(
    "Historical portfolio risk analysis using locally available price files. "
    "Equal weighting is used for this educational simulation."
)

risk_master_file = os.path.join(
    DATA_PATH, "synchronized_market_data.csv"
)

if os.path.exists(risk_master_file):

    risk_master = pd.read_csv(risk_master_file)

    if not risk_master.empty and "symbol" in risk_master.columns:

        available_symbols = sorted(
            risk_master["symbol"].dropna().astype(str).unique().tolist()
        )

        selected_symbols = st.multiselect(
            "Select securities for portfolio analysis",
            available_symbols,
            default=available_symbols[:3],
            max_selections=20,
            key="portfolio_risk_symbols"
        )

        if st.button(
            "Calculate Portfolio Risk",
            key="calculate_portfolio_risk"
        ):

            if len(selected_symbols) < 2:
                st.warning("Select at least two securities.")
            else:
                price_series = {}

                for symbol in selected_symbols:

                    match = risk_master[
                        risk_master["symbol"].astype(str) == symbol
                    ]

                    if match.empty:
                        continue

                    file_key = str(match.iloc[0]["file_key"])
                    price_file = os.path.join(
                        PRICE_PATH, f"{file_key}.csv"
                    )

                    if not os.path.exists(price_file):
                        continue

                    try:
                        price_df = pd.read_csv(price_file)

                        date_col = next(
                            (
                                col for col in price_df.columns
                                if str(col).lower() in ["date", "datetime"]
                            ),
                            price_df.columns[0]
                        )

                        close_col = next(
                            (
                                col for col in ["Close", "Adj Close", "close"]
                                if col in price_df.columns
                            ),
                            None
                        )

                        if close_col is None:
                            continue

                        dates = pd.to_datetime(
                            price_df[date_col], errors="coerce"
                        )

                        closes = pd.to_numeric(
                            price_df[close_col], errors="coerce"
                        )

                        series = pd.Series(
                            closes.values, index=dates
                        ).dropna()

                        series = series[
                            ~series.index.isna()
                        ]

                        series = series[
                            ~series.index.duplicated(keep="last")
                        ].sort_index()

                        if len(series) >= 30:
                            price_series[symbol] = series

                    except Exception:
                        continue

                if len(price_series) < 2:
                    st.warning(
                        "Insufficient local historical price files. "
                        "At least two securities need 30 or more observations."
                    )

                else:
                    prices = pd.concat(
                        price_series,
                        axis=1,
                        join="inner"
                    ).dropna()

                    prices = prices[
                        (prices > 0).all(axis=1)
                    ]

                    if len(prices) < 30:
                        st.warning(
                            "Insufficient common historical dates "
                            "to calculate portfolio risk."
                        )
                    else:
                        returns = prices.pct_change().dropna()

                        weights = np.repeat(
                            1 / len(returns.columns),
                            len(returns.columns)
                        )

                        portfolio_returns = returns.dot(weights)

                        cumulative = (
                            1 + portfolio_returns
                        ).cumprod()

                        running_peak = cumulative.cummax()

                        drawdown = (
                            cumulative / running_peak - 1
                        ) * 100

                        total_return = (
                            cumulative.iloc[-1] - 1
                        ) * 100

                        annual_volatility = (
                            portfolio_returns.std() * np.sqrt(252) * 100
                        )

                        annual_return = (
                            cumulative.iloc[-1] **
                            (252 / len(portfolio_returns)) - 1
                        ) * 100

                        sharpe = (
                            portfolio_returns.mean()
                            / portfolio_returns.std()
                            * np.sqrt(252)
                            if portfolio_returns.std() > 0
                            else np.nan
                        )

                        daily_var = (
                            portfolio_returns.quantile(0.05) * 100
                        )

                        max_drawdown = drawdown.min()

                        st.success(
                            f"Analysis completed using {len(prices.columns)} "
                            f"securities and {len(returns)} common daily returns."
                        )

                        a, b, c = st.columns(3)

                        a.metric(
                            "Historical Total Return",
                            f"{total_return:.2f}%"
                        )

                        b.metric(
                            "Annualized Volatility",
                            f"{annual_volatility:.2f}%"
                        )

                        c.metric(
                            "Maximum Drawdown",
                            f"{max_drawdown:.2f}%"
                        )

                        d, e, f = st.columns(3)

                        d.metric(
                            "Annualized Return",
                            f"{annual_return:.2f}%"
                        )

                        e.metric(
                            "Sharpe Ratio",
                            f"{sharpe:.2f}"
                            if pd.notna(sharpe) else "Unavailable"
                        )

                        f.metric(
                            "Daily 95% Historical VaR",
                            f"{daily_var:.2f}%"
                        )

                        st.subheader("Portfolio Growth")

                        growth_fig = go.Figure()

                        growth_fig.add_trace(
                            go.Scatter(
                                x=cumulative.index,
                                y=(cumulative - 1) * 100,
                                mode="lines",
                                name="Portfolio Return"
                            )
                        )

                        growth_fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Cumulative Return (%)",
                            height=400
                        )

                        st.plotly_chart(
                            growth_fig,
                            use_container_width=True
                        )

                        st.subheader("Historical Drawdown")

                        dd_fig = go.Figure()

                        dd_fig.add_trace(
                            go.Scatter(
                                x=drawdown.index,
                                y=drawdown.values,
                                fill="tozeroy",
                                name="Drawdown (%)"
                            )
                        )

                        dd_fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Drawdown (%)",
                            height=350
                        )

                        st.plotly_chart(
                            dd_fig,
                            use_container_width=True
                        )

                        st.subheader("Portfolio Constituents")

                        constituent_df = pd.DataFrame({
                            "Symbol": returns.columns,
                            "Equal Weight (%)": [
                                round(100 / len(returns.columns), 2)
                            ] * len(returns.columns),
                            "Observations": [
                                len(returns)
                            ] * len(returns.columns)
                        })

                        st.dataframe(
                            constituent_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download Portfolio Daily Returns",
                            data=portfolio_returns.to_csv(
                                header=["portfolio_daily_return"],
                                index_label="date"
                            ),
                            file_name="portfolio_risk_returns.csv",
                            mime="text/csv"
                        )

                        st.caption(
                            "Risk metrics are based on historical equal-weighted "
                            "returns. They do not predict future performance. "
                            "VaR is a historical 95% daily estimate."
                        )

else:
    st.warning("Synchronized market dataset not found.")

# PORTFOLIO_RISK_ANALYTICS_END



# DATA_QUALITY_INTELLIGENCE_START

st.divider()
st.header("🛡️ Data Quality & Coverage Intelligence")

st.caption(
    "Data validation, listing coverage, missing values and historical "
    "price-file health. Results describe available project data only."
)

try:
    from data_quality import (
        audit_dataframe,
        column_quality,
        coverage_table,
        price_file_audit,
        quality_score
    )

    dq_master = pd.read_csv(MASTER_FILE)
    dq_market_path = os.path.join(
        DATA_PATH, "synchronized_market_data.csv"
    )

    if os.path.exists(dq_market_path):
        dq_market = pd.read_csv(dq_market_path)

        dq_score = quality_score(dq_market, dq_master)
        dq_audit = audit_dataframe(dq_market)
        dq_missing = column_quality(dq_market)
        dq_country = coverage_table(
            dq_master, dq_market, "country"
        )
        dq_exchange = coverage_table(
            dq_master, dq_market, "exchange_code"
        )
        dq_files = price_file_audit(PRICE_PATH)

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Quality Indicator", f"{dq_score}/100")
        q2.metric("Priced Securities", f"{len(dq_market):,}")
        q3.metric("Missing Cells", f"{dq_audit['missing_cells']:,}")
        q4.metric("Price Files", f"{len(dq_files):,}")

        st.subheader("1. Market Data Audit")
        st.json(dq_audit)

        st.subheader("2. Missing Values by Column")
        st.dataframe(
            dq_missing,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("3. Country Coverage")
        st.dataframe(
            dq_country,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("4. Exchange Coverage")
        st.dataframe(
            dq_exchange,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("5. Price File Health")
        st.dataframe(
            dq_files.head(100),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download Missing Value Report",
            dq_missing.to_csv(index=False),
            file_name="data_quality_missing_values.csv",
            mime="text/csv"
        )

        st.download_button(
            "📥 Download Country Coverage",
            dq_country.to_csv(index=False),
            file_name="data_quality_country_coverage.csv",
            mime="text/csv"
        )

        st.download_button(
            "📥 Download Exchange Coverage",
            dq_exchange.to_csv(index=False),
            file_name="data_quality_exchange_coverage.csv",
            mime="text/csv"
        )

    else:
        st.warning("Synchronized market data is not available.")

except Exception as dq_error:
    st.error(f"Data Quality module error: {dq_error}")

# DATA_QUALITY_INTELLIGENCE_END


# ETF_ANALYSIS_SECTION_START

st.divider()
st.header("📊 ETF Holdings & Comparison Lab")

st.caption(
    "Educational ETF research using available Yahoo Finance data. "
    "Unavailable provider data is not estimated."
)

from etf_analysis import (
    get_etf_analysis,
    compare_etfs,
    calculate_holdings_overlap
)

etf_input = st.text_input(
    "Enter ETF symbols separated by commas",
    value="AAAP, SPY",
    key="etf_analysis_symbols"
)

etf_symbols = list(dict.fromkeys(
    x.strip().upper()
    for x in etf_input.split(",")
    if x.strip()
))

if st.button("🔍 Analyse ETFs", key="run_etf_analysis"):

    if not etf_symbols:
        st.warning("Enter at least one ETF symbol.")

    else:
        with st.spinner("Fetching ETF research data..."):

            etf_results = {}

            for symbol in etf_symbols:
                etf_results[symbol] = get_etf_analysis(symbol)

            st.subheader("1. ETF Comparison")

            comparison = compare_etfs(etf_symbols)

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Download ETF Comparison",
                comparison.to_csv(index=False),
                "etf_comparison.csv",
                "text/csv",
                key="download_etf_comparison"
            )

            st.divider()
            st.subheader("2. Individual ETF Analysis")

            for symbol, data in etf_results.items():

                with st.expander(
                    f"{symbol} — Holdings & Fund Profile",
                    expanded=True
                ):

                    st.markdown("**Top Holdings**")

                    holdings = data["top_holdings"]

                    if not holdings.empty:
                        st.dataframe(
                            holdings,
                            use_container_width=True
                        )
                    else:
                        st.info(
                            "Top holdings unavailable from provider."
                        )

                    st.markdown("**Sector Allocation**")

                    sectors = {
                        k: v for k, v in
                        data["sector_weightings"].items()
                        if v is not None
                    }

                    if sectors:
                        sector_df = pd.DataFrame({
                            "Sector": list(sectors.keys()),
                            "Weight (%)": list(sectors.values())
                        })

                        st.bar_chart(
                            sector_df.set_index("Sector")
                        )
                    else:
                        st.info("Sector allocation unavailable.")

                    st.markdown("**Asset Allocation**")

                    assets = {
                        k: v for k, v in
                        data["asset_classes"].items()
                        if v is not None
                    }

                    if assets:
                        asset_df = pd.DataFrame({
                            "Asset Class": list(assets.keys()),
                            "Weight (%)": list(assets.values())
                        })

                        st.dataframe(
                            asset_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        positive_assets = {
                            k: v for k, v in assets.items()
                            if v > 0
                        }

                        if positive_assets:
                            st.bar_chart(
                                pd.DataFrame.from_dict(
                                    positive_assets,
                                    orient="index",
                                    columns=["Weight (%)"]
                                )
                            )
                    else:
                        st.info("Asset allocation unavailable.")

                    st.markdown("**Bond Profile**")

                    bonds = data["bond_holdings"]

                    if not bonds.empty:
                        st.dataframe(
                            bonds,
                            use_container_width=True
                        )
                    else:
                        st.info(
                            "Detailed bond profile unavailable."
                        )

                    st.caption(
                        "Provider status: " + str(data["status"])
                    )

            st.divider()
            st.subheader("3. Holdings Overlap")

            if len(etf_symbols) >= 2:

                overlap = calculate_holdings_overlap(
                    etf_results[etf_symbols[0]],
                    etf_results[etf_symbols[1]]
                )

                st.info(overlap["message"])

                if not overlap["overlap"].empty:
                    st.dataframe(
                        overlap["overlap"],
                        use_container_width=True
                    )

            else:
                st.info(
                    "Enter at least two ETFs to calculate overlap."
                )

# ETF_ANALYSIS_SECTION_END


# Technical screener
st.subheader("⚙️ Technical Screener")

screener = load_screener()

if not screener.empty:
    st.dataframe(
        screener,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No screener results available yet.")

st.caption(
    "GlobalStock AI | Research and educational analytics platform"
)


# ADVANCED_RESEARCH_LAB_START

st.markdown("---")
st.header("🔬 Advanced Research & Risk Analytics")

st.caption(
    "Historical stock-return correlation, performance and risk metrics. "
    "Educational research only."
)

from advanced_analytics import (
    list_available_stocks,
    load_close_prices,
    calculate_risk_metrics,
    calculate_correlation,
    normalized_performance,
    rolling_volatility
)

advanced_stocks = list_available_stocks(
    MASTER_FILE,
    PRICE_PATH
)

if advanced_stocks.empty:

    st.warning("No supported price files found.")

else:

    countries = sorted(
        advanced_stocks["country"].dropna().unique()
    )

    selected_country = st.selectbox(
        "Research Country",
        ["All Countries"] + countries,
        key="advanced_research_country"
    )

    filtered_stocks = advanced_stocks.copy()

    if selected_country != "All Countries":
        filtered_stocks = filtered_stocks[
            filtered_stocks["country"] == selected_country
        ]

    label_map = {}

    for _, row in filtered_stocks.iterrows():

        label = (
            str(row["company_name"])
            + " ("
            + str(row["symbol"])
            + ")"
        )

        label_map[label] = row["file_key"]

    labels = list(label_map.keys())

    selected_labels = st.multiselect(
        "Select stocks (maximum 12)",
        labels,
        default=labels[:min(5, len(labels))],
        max_selections=12,
        key="advanced_research_stocks"
    )

    if selected_labels:

        selected_keys = [
            label_map[label]
            for label in selected_labels
        ]

        prices = load_close_prices(
            selected_keys,
            PRICE_PATH
        )

        if not prices.empty:

            st.subheader("Historical Risk Metrics")

            metrics = calculate_risk_metrics(prices)

            st.dataframe(
                metrics,
                use_container_width=True
            )

            st.download_button(
                "Download Risk Metrics",
                metrics.to_csv(index=False),
                "risk_metrics.csv",
                "text/csv"
            )

            st.subheader("Historical Performance")

            normalized = normalized_performance(prices)

            st.line_chart(normalized)

            st.subheader("Return Correlation")

            if len(prices.columns) >= 2:

                correlation = calculate_correlation(prices)

                fig = go.Figure(
                    data=go.Heatmap(
                        z=correlation.values,
                        x=correlation.columns,
                        y=correlation.index,
                        zmin=-1,
                        zmax=1,
                        colorscale="RdBu"
                    )
                )

                fig.update_layout(
                    title="Historical Return Correlation",
                    height=550
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.download_button(
                    "Download Correlation Matrix",
                    correlation.to_csv(),
                    "correlation_matrix.csv",
                    "text/csv"
                )

            else:
                st.info(
                    "Select at least two stocks for correlation analysis."
                )

            st.subheader("20-Day Rolling Volatility")

            rolling = rolling_volatility(prices)

            st.line_chart(rolling)

            st.download_button(
                "Download Normalized Performance",
                normalized.to_csv(),
                "normalized_performance.csv",
                "text/csv"
            )

            st.info(
                "Metrics are calculated from available historical data. "
                "Different trading calendars, missing observations and "
                "currencies can affect comparisons. Historical results "
                "do not predict future performance."
            )

        else:
            st.warning("Could not load sufficient historical price data.")

    else:
        st.info("Select stocks to begin research.")

# ADVANCED_RESEARCH_LAB_END



# DATA_QUALITY_START

st.markdown("---")
st.header("🛡️ Global Data Quality Monitor")

st.caption(
    "Historical dataset completeness, integrity and coverage monitoring."
)

from data_quality import build_quality_report

quality_dir = os.path.join(
    PROJECT_PATH,
    "data",
    "processed",
    "data_quality"
)

quality_file = os.path.join(
    quality_dir,
    "price_quality_report.csv"
)

coverage_file = os.path.join(
    quality_dir,
    "global_coverage_report.csv"
)

country_file = os.path.join(
    quality_dir,
    "country_quality_report.csv"
)

exchange_file = os.path.join(
    quality_dir,
    "exchange_quality_report.csv"
)

if all(os.path.exists(p) for p in [
    quality_file, coverage_file,
    country_file, exchange_file
]):

    quality_data = pd.read_csv(quality_file)
    coverage_data = pd.read_csv(coverage_file)
    country_data = pd.read_csv(country_file)
    exchange_data = pd.read_csv(exchange_file)

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Price Files",
        len(quality_data)
    )

    q2.metric(
        "Inspected Successfully",
        int((quality_data["status"] == "OK").sum())
    )

    q3.metric(
        "Average Quality Score",
        round(quality_data["quality_score"].mean(), 2)
    )

    st.subheader("Country-wise Coverage")

    st.dataframe(
        country_data,
        use_container_width=True
    )

    st.subheader("Exchange-wise Coverage")

    st.dataframe(
        exchange_data,
        use_container_width=True
    )

    st.subheader("Data Quality Issues")

    issue_columns = [
        "file", "missing_values",
        "duplicate_dates", "invalid_ohlc",
        "negative_volume", "missing_dates"
    ]

    st.dataframe(
        quality_data[issue_columns],
        use_container_width=True
    )

    st.download_button(
        "Download Complete Quality Report",
        quality_data.to_csv(index=False),
        "price_quality_report.csv",
        "text/csv"
    )

else:

    st.warning(
        "Quality reports not found. Run the Data Quality Engine first."
    )

# DATA_QUALITY_END
