#!/usr/bin/env python3
"""
Source Market Data Script

Overview:
This script is responsible for fetching raw financial market data required for the
SEC Risk Factor Quantitative Framework. It sources data for specified company
tickers, market benchmarks, risk-free rates, and optionally, common risk factors
(e.g., Fama-French). The fetched data is saved to a structured directory for
later processing and integration with NLP features.

This script operates independently of `sec_ingestion.py` and `feature_engineer_nlp.py`
in terms of direct file inputs. Its sole responsibility is to gather raw market data.

IMPORTANT: The list of `COMPANIES_TO_PROCESS` defined in this script should be
consistent with the tickers processed by `sec_ingestion.py` and subsequently
`feature_engineer_nlp.py`. In a production-grade pipeline, this list would ideally
be loaded from a shared configuration file (e.g., config/parameters.yml) to ensure
all pipeline stages operate on the same set of entities.

The output data is saved as Parquet files. Time-series data (daily stock prices,
benchmark, risk-free rate) will have a DatetimeIndex, which is crucial for
accurate temporal alignment in subsequent data integration steps.

Core Functionality:
1.  **Configuration:**
    -   Defines tickers, date ranges, and data sources (initially yfinance).
    -   Future: Integrate with `config/parameters.yml` for shared configurations.
2.  **Data Fetching:**
    -   Downloads daily stock data (OHLC, Adjusted Close, Volume).
    -   Downloads market benchmark data (e.g., S&P 500).
    -   Downloads risk-free rate data (e.g., Treasury yields).
    -   (Placeholder) Downloads Fama-French factor data.
3.  **Data Storage:**
    -   Saves fetched data into a `data/raw/market_data/` directory structure,
        typically as Parquet files.
4.  **Logging:**
    -   Logs progress, errors, and summaries of downloaded data.
"""

import logging
import time
from pathlib import Path
import pandas as pd
import yfinance as yf
# import yaml # For loading config from parameters.yml in the future

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        # Consider adding a FileHandler for this script's specific logs
        # logging.FileHandler("source_market_data.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

# --- Configuration ---
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path(".") # Fallback for interactive use/testing

RAW_MARKET_DATA_DIR = BASE_DIR / "data" / "raw" / "market_data"
DAILY_STOCK_DIR = RAW_MARKET_DATA_DIR / "daily_stock"
BENCHMARK_DIR = RAW_MARKET_DATA_DIR / "benchmark"
RISK_FREE_RATE_DIR = RAW_MARKET_DATA_DIR / "risk_free_rate"
FACTORS_DIR = RAW_MARKET_DATA_DIR / "factors"

# Create directories if they don't exist
DAILY_STOCK_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
RISK_FREE_RATE_DIR.mkdir(parents=True, exist_ok=True)
FACTORS_DIR.mkdir(parents=True, exist_ok=True)

# --- Parameters (Future: Load from config/parameters.yml) ---
# IMPORTANT: This list should be consistent with the companies processed by sec_ingestion.py.
# The list provided in sec_ingestion.py was:
# companies = [
#     {"ticker": "NVDA", "name": "NVIDIA Corporation"},
#     {"ticker": "MSFT", "name": "Microsoft Corporation"},
#     {"ticker": "META", "name": "Meta Platforms, Inc."},
#     {"ticker": "TGT",  "name": "Target Corporation"},
#     {"ticker": "JPM",  "name": "JPMorgan Chase & Co."},
#     {"ticker": "SCHW", "name": "Charles Schwab Corporation"},
#     {"ticker": "XOM",  "name": "Exxon Mobil Corporation"},
# ]
# This `COMPANIES_TO_PROCESS` list should match the 'ticker' values from above.
COMPANIES_TO_PROCESS = ["NVDA", "MSFT", "META", "TGT", "JPM", "SCHW", "XOM"]

# Define the overall period for which data might be needed.
# This should be broad enough to cover filings and lookback/forward windows.
START_DATE = "2008-01-01"
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d') # Today's date

# Market benchmark ticker
BENCHMARK_TICKER = "^GSPC" # S&P 500 Index
# Risk-free rate proxy
RISK_FREE_TICKER = "^IRX" # 13 Week Treasury Bill
# --- End Configuration ---


def fetch_daily_stock_data(tickers: list, start_date: str, end_date: str, output_dir: Path) -> None:
    """
    Fetches daily OHLC, Adjusted Close, and Volume for a list of stock tickers
    using yfinance and saves each to a Parquet file. The resulting Parquet files
    will have a DatetimeIndex, suitable for temporal alignment.

    Args:
        tickers: A list of stock ticker symbols.
        start_date: The start date for data fetching (YYYY-MM-DD).
        end_date: The end date for data fetching (YYYY-MM-DD).
        output_dir: The directory to save the Parquet files.
    """
    logger.info(f"Fetching daily stock data for {len(tickers)} tickers from {start_date} to {end_date}.")
    for ticker_symbol in tickers:
        try:
            logger.info(f"Fetching data for {ticker_symbol}...")
            # Download data using yfinance
            # 'actions=True' includes dividends and stock splits, which are reflected in 'Adj Close'.
            # The index of the returned DataFrame is a DatetimeIndex.
            data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False, actions=True)

            if data.empty:
                logger.warning(f"No data found for {ticker_symbol} for the given period.")
                continue

            # Ensure column names are suitable for Parquet and general use (e.g., no spaces)
            data.columns = [col.replace(" ", "_") for col in data.columns]

            # Save to Parquet. The DatetimeIndex will be preserved.
            file_path = output_dir / f"{ticker_symbol}.parquet"
            data.to_parquet(file_path)
            logger.info(f"Saved data for {ticker_symbol} to {file_path}")
            time.sleep(0.5) # Small polite delay to avoid overwhelming the API

        except Exception as e:
            logger.error(f"Failed to fetch or save data for {ticker_symbol}: {e}", exc_info=True)
    logger.info("Finished fetching daily stock data.")


def fetch_benchmark_data(benchmark_ticker: str, start_date: str, end_date: str, output_dir: Path) -> None:
    """
    Fetches daily data for a market benchmark (e.g., S&P 500) using yfinance.
    The resulting Parquet file will have a DatetimeIndex.

    Args:
        benchmark_ticker: The ticker symbol for the benchmark (e.g., "^GSPC").
        start_date: The start date for data fetching (YYYY-MM-DD).
        end_date: The end date for data fetching (YYYY-MM-DD).
        output_dir: The directory to save the Parquet file.
    """
    logger.info(f"Fetching benchmark data for {benchmark_ticker} from {start_date} to {end_date}.")
    try:
        data = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            logger.warning(f"No data found for benchmark {benchmark_ticker}.")
            return

        data.columns = [col.replace(" ", "_") for col in data.columns]
        # The index is already a DatetimeIndex from yfinance.
        file_path = output_dir / f"{benchmark_ticker.replace('^','')}.parquet" # Remove ^ for cleaner filename
        data.to_parquet(file_path)
        logger.info(f"Saved benchmark data for {benchmark_ticker} to {file_path}")

    except Exception as e:
        logger.error(f"Failed to fetch or save benchmark data for {benchmark_ticker}: {e}", exc_info=True)
    logger.info("Finished fetching benchmark data.")


def fetch_risk_free_rate_data(risk_free_ticker: str, start_date: str, end_date: str, output_dir: Path) -> None:
    """
    Fetches data for a risk-free rate proxy (e.g., Treasury yield) using yfinance.
    The resulting Parquet file will have a DatetimeIndex.
    Note: yfinance data for Treasury yields can be limited or less reliable.
    Consider alternative sources like FRED (Federal Reserve Economic Data) for more robust data.

    Args:
        risk_free_ticker: The ticker symbol for the risk-free rate proxy (e.g., "^IRX").
        start_date: The start date for data fetching (YYYY-MM-DD).
        end_date: The end date for data fetching (YYYY-MM-DD).
        output_dir: The directory to save the Parquet file.
    """
    logger.info(f"Fetching risk-free rate data for {risk_free_ticker} from {start_date} to {end_date}.")
    try:
        # yfinance provides yields as percentages, so 'Close' price is the yield.
        data = yf.download(risk_free_ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            logger.warning(f"No data found for risk-free rate proxy {risk_free_ticker}.")
            return

        data.columns = [col.replace(" ", "_") for col in data.columns]
        # We are interested in the 'Close' price which represents the yield.
        # Convert to decimal if it's a percentage for easier use (e.g., 1.5% -> 0.015)
        if 'Close' in data.columns: # Ensure 'Close' column exists
            data['Risk_Free_Rate'] = data['Close'] / 100.0 # Assuming yfinance returns it as percentage points
            # Select and rename relevant columns for clarity, keeping the DatetimeIndex
            data = data[['Risk_Free_Rate']].copy()
        else:
            logger.warning(f"'Close' column not found for {risk_free_ticker}. Cannot calculate Risk_Free_Rate.")
            return

        file_path = output_dir / f"{risk_free_ticker.replace('^','')}_rate.parquet"
        data.to_parquet(file_path)
        logger.info(f"Saved risk-free rate data for {risk_free_ticker} to {file_path}")

    except Exception as e:
        logger.error(f"Failed to fetch or save risk-free rate data for {risk_free_ticker}: {e}", exc_info=True)
    logger.info("Finished fetching risk-free rate data.")

def fetch_fama_french_factors(start_date: str, end_date: str, output_dir: Path) -> None:
    """
    Placeholder function to fetch Fama-French factors.
    This typically involves downloading a CSV/zip from Kenneth French's website
    and parsing it. pandas_datareader also has some capability.
    The data should be processed to have a DatetimeIndex.

    Args:
        start_date: The start date for data (YYYY-MM-DD or YYYYMM).
        end_date: The end date for data (YYYY-MM-DD or YYYYMM).
        output_dir: The directory to save the Parquet file.
    """
    logger.info("Attempting to fetch Fama-French factors (placeholder).")
    # Example using pandas_datareader (requires internet and might change):
    # Ensure you have pandas_datareader installed: pip install pandas-datareader
    # try:
    #     from pandas_datareader.famafrench import get_available_datasets
    #     import pandas_datareader.data as web
    #     logger.info(f"Available Fama-French datasets from pandas_datareader: {get_available_datasets()}")
    #     # Example: Fetch F-F_Research_Data_5_Factors_2x3_daily
    #     # The dates for Fama-French are often end-of-month or specific formats.
    #     # Adjust start_date/end_date format or parsing as needed.
    #     # Using an example date range. pandas_datareader might handle string dates.
    #     ff_factors_data = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start=start_date, end=end_date)
    #     # ff_factors_data is a dictionary of DataFrames. Select the relevant one (usually index 0 for daily/monthly).
    #     if ff_factors_data and 0 in ff_factors_data:
    #         factors_df = ff_factors_data[0].copy()
    #         # Convert percentage values to decimals (e.g., 1.0 means 1%)
    #         factors_df = factors_df / 100.0
    #         # The index from pandas_datareader for Fama-French should already be a DatetimeIndex or PeriodIndex.
    #         # If it's PeriodIndex, convert to DatetimeIndex:
    #         # if isinstance(factors_df.index, pd.PeriodIndex):
    #         #     factors_df.index = factors_df.index.to_timestamp()
    #         file_path = output_dir / "fama_french_5_factors_daily.parquet"
    #         factors_df.to_parquet(file_path)
    #         logger.info(f"Saved Fama-French factors to {file_path}")
    #     else:
    #         logger.warning("Could not retrieve Fama-French factors using pandas_datareader.")
    # except Exception as e_ff:
    #     logger.error(f"Error fetching Fama-French factors: {e_ff}")

    logger.warning("Fama-French factor fetching is not fully implemented in this template.")
    logger.warning("You'll need to implement the download and parsing logic, e.g., using pandas_datareader or direct download from Kenneth French's website.")
    # For now, let's create a dummy file to indicate where it would go.
    dummy_df = pd.DataFrame({'Mkt-RF': [0.0], 'SMB': [0.0], 'HML': [0.0], 'RMW': [0.0], 'CMA': [0.0]},
                            index=pd.to_datetime([pd.Timestamp(start_date)]))
    dummy_df.index.name = 'Date' # Ensure the index has a name, common for time series.
    file_path = output_dir / "fama_french_factors_placeholder.parquet"
    try:
        dummy_df.to_parquet(file_path)
        logger.info(f"Saved placeholder Fama-French factors to {file_path}")
    except Exception as e:
        logger.error(f"Could not save placeholder Fama-French factors: {e}")


# =============================================================================
# Main Orchestration Function
# =============================================================================

def main():
    """
    Main orchestration function to fetch all required market data.
    """
    start_time_main = time.time()
    logger.info("--- Starting Market Data Sourcing Pipeline ---")

    # 1. Fetch Daily Stock Data
    fetch_daily_stock_data(COMPANIES_TO_PROCESS, START_DATE, END_DATE, DAILY_STOCK_DIR)

    # 2. Fetch Market Benchmark Data
    fetch_benchmark_data(BENCHMARK_TICKER, START_DATE, END_DATE, BENCHMARK_DIR)

    # 3. Fetch Risk-Free Rate Data
    fetch_risk_free_rate_data(RISK_FREE_TICKER, START_DATE, END_DATE, RISK_FREE_RATE_DIR)

    # 4. Fetch Fama-French Factor Data (Optional)
    fetch_fama_french_factors(START_DATE, END_DATE, FACTORS_DIR)

    end_time_main = time.time()
    logger.info(f"--- Market Data Sourcing Pipeline Finished ---")
    logger.info(f"Total execution time: {end_time_main - start_time_main:.2f} seconds")

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
