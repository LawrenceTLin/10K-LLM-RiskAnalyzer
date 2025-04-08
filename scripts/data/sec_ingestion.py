#!/usr/bin/env python3
"""
SEC 10-K Risk Factor Extractor (API-Focused)

Relies on sec-api.io Query and Extractor APIs to retrieve risk factors.
Includes minimal scraping only to find the document URL from the index page.
"""

import os
import re
import json
import time
import logging
import requests # Still needed for throttled_request used by scraping index search
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup # Still needed for index page parsing
from urllib.parse import urlparse, quote

# Enhanced dependencies
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type # Still needed for throttled_request
# from unstructured.partition.html import partition_html # REMOVED
from tqdm import tqdm # Keep if using process_multiple_companies
from dotenv import load_dotenv
import asyncio
import httpx
from aiolimiter import AsyncLimiter

# --- Conditional Import for QueryApi ---
try:
    from sec_api import QueryApi
    SEC_QUERY_API_AVAILABLE = True
except ImportError:
    SEC_QUERY_API_AVAILABLE = False
    QueryApi = None
    print("WARNING: sec-api library not found or QueryApi missing. API Query features unavailable.")
# --- End Conditional Import ---

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, # Changed back to INFO for less noise, set DEBUG if needed
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("sec_api_extractor.log", mode='w') # Changed log file name
    ]
)
logger = logging.getLogger(__name__)
# --- End Logging Config ---

# --- Data Validation and Storage ---
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import schema as pa_schema, Table
import pandera as pda # Correct alias
from pandera import Column, DataFrameSchema, Check, dtypes
# --- End Data Validation ---

# --- Optional NLP Enhancements ---
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
# --- End NLP ---

# --- Environment Variables & Setup ---
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "sec_filings"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed" / "sec_filings"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis" / "risk_factors"

for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, ANALYSIS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
# --- End Setup ---

# --- Constants ---
SEC_BASE_URL = "https://www.sec.gov"
SEC_ARCHIVES_URL = f"{SEC_BASE_URL}/Archives"
USER_AGENT = "RiskFactorAnalysisResearch ltl32@cornell.edu" # Use your email/contact
MAX_API_CALLS = 100
# --- End Constants ---

# --- Pandera and PyArrow Schemas ---
risk_factor_schema = DataFrameSchema({
    "ticker": Column(str, Check.str_length(min_value=1, max_value=5)),
    "year": Column(int, Check.in_range(1993, datetime.now().year + 1)), # Allow current year
    "cik": Column(str, nullable=True),
    "paragraph_idx": Column(int),
    "sentence_idx": Column(int),
    "sentence": Column(str, Check.str_length(min_value=5)), # Reduced min length slightly
    "word_count": Column(int, Check.greater_than_or_equal_to(1)), # Allow 1 word
    "contains_risk": Column(bool),
    "contains_uncertainty": Column(bool),
    "sentiment_score": Column(dtypes.Float32, Check.in_range(-1, 1), nullable=True),
    "risk_keywords": Column(list, nullable=True)
})

arrow_schema = pa_schema([
    ("ticker", pa.string()), ("year", pa.int16()), ("cik", pa.string()),
    ("paragraph_idx", pa.int32()), ("sentence_idx", pa.int32()),
    ("sentence", pa.string()), ("word_count", pa.int32()),
    ("contains_risk", pa.bool_()), ("contains_uncertainty", pa.bool_()),
    ("sentiment_score", pa.float32()), ("risk_keywords", pa.list_(pa.string()))
])
# --- End Schemas ---


# =============================================================================
# NLP Processing Class
# =============================================================================
class RiskVectorizer:
    """Process text with NLP models"""
    def __init__(self, model_name="ProsusAI/finbert"):
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers lib not available. NLP features disabled.")
            self.model = None; self.tokenizer = None; return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info(f"Loaded NLP model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load NLP model '{model_name}': {e}")
            self.model = None; self.tokenizer = None

    def vectorize_sentence(self, text: str) -> tuple[float, str]:
        if not self.model or not self.tokenizer or not text: return 0.0, "neutral"
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            probs = outputs.logits.softmax(dim=1).detach().numpy()[0]
            score = float(probs[2] - probs[0]) # Ensure Python float
            label = "negative" if probs[0] > probs[1] and probs[0] > probs[2] else \
                    "positive" if probs[2] > probs[1] and probs[2] > probs[0] else "neutral"
            return score, label
        except Exception as e:
            logger.error(f"Error in sentiment analysis for text (len={len(text)}): {e}", exc_info=False) # Avoid logging full text
            return 0.0, "neutral"

    def extract_risk_keywords(self, text: str) -> List[str]:
        if not text: return []
        risk_terms = { # Use set for faster lookups
            "risk", "uncertainty", "adverse", "negative", "volatility", "decline",
            "failure", "litigation", "regulatory", "competitive", "liability",
            "disruption", "fluctuation", "economic", "pandemic", "recession",
            "inflationary", "cybersecurity", "breach", "compliance", "delay",
            "shortage", "downturn", "debt", "lawsuit"}
        found_terms = set()
        # Simple word check, could be improved with stemming/lemmatization
        text_lower_words = set(re.findall(r'\b\w+\b', text.lower()))
        found_terms = list(risk_terms.intersection(text_lower_words))
        return found_terms

# =============================================================================
# Asynchronous SEC Interaction Class
# =============================================================================
class AsyncSECProcessor:
    """Asynchronous processor for SEC interactions (API and generic fetch)."""
    def __init__(self, api_key: Optional[str] = None, requests_per_second: int = 10):
        self.api_key = api_key
        self.limiter = AsyncLimiter(requests_per_second, 1)
        logger.info(f"Initialized AsyncSECProcessor with rate limit: {requests_per_second} req/sec.")
        if not self.api_key: logger.warning("No API key provided. Direct Extractor API calls will fail.")

    async def fetch_section_via_extractor_http(self, document_url: str, item: str = "1A", return_type: str = "text") -> Optional[Dict[str, Any]]:
        """Fetches a section using the direct HTTP GET /extractor endpoint."""
        if not self.api_key: logger.warning("[Extractor API] No API key."); return None
        if not document_url: logger.warning("[Extractor API] No document_url."); return None
        try:
            encoded_doc_url = quote(document_url, safe='')
            target_url = f"https://api.sec-api.io/extractor?url={encoded_doc_url}&item={item}&type={return_type}&token={self.api_key}"
            logger.debug(f"[Extractor API] Requesting item '{item}' for URL ending: ...{document_url[-60:]}")
        except Exception as url_e:
            logger.error(f"[Extractor API] Error constructing URL: {url_e}"); return None

        async with self.limiter:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(target_url, timeout=60.0)
                if response.status_code == 200:
                    content = response.text
                    if content:
                        logger.info(f"[Extractor API] Success (200 OK) for item '{item}', URL ending: ...{document_url[-60:]}")
                        return {"content": content}
                    else:
                        logger.warning(f"[Extractor API] Success (200 OK) but empty content for item '{item}', URL ending: ...{document_url[-60:]}")
                        return None
                elif response.status_code == 404:
                    logger.warning(f"[Extractor API] Failed (404 Not Found) for item '{item}', URL ending: ...{document_url[-60:]}")
                elif response.status_code in [401, 403]:
                    logger.error(f"[Extractor API] Failed ({response.status_code} Unauthorized/Forbidden). Check API key.")
                elif response.status_code == 400:
                    logger.error(f"[Extractor API] Failed ({response.status_code} Bad Request). Check params. Response: {response.text[:200]}")
                else:
                    logger.error(f"[Extractor API] Failed ({response.status_code}). URL ending: ...{document_url[-60:]}. Response: {response.text[:200]}")
                return None # Return None for all non-200 responses after logging
            except httpx.TimeoutException: logger.error(f"[Extractor API] Request timed out."); return None
            except httpx.RequestError as e: logger.error(f"[Extractor API] Request error: {e}"); return None
            except Exception as e: logger.error(f"[Extractor API] Unexpected error: {e}", exc_info=True); return None

    async def fetch_filing(self, url: str, headers: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Fetches a generic filing document (HTML)."""
        if headers is None:
            headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"}
            try:
                parsed_url = urlparse(url);
                if parsed_url.netloc: headers["Host"] = parsed_url.netloc
            except Exception: pass
        async with self.limiter:
            try:
                async with httpx.AsyncClient() as client:
                    logger.debug(f"[Generic Fetch] Fetching URL: {url}")
                    response = await client.get(url, headers=headers, timeout=30.0, follow_redirects=True)
                response.raise_for_status() # Check for 4xx/5xx errors
                logger.debug(f"[Generic Fetch] Success ({response.status_code}) for: {url}")
                return response.text
            except httpx.HTTPStatusError as e: logger.error(f"[Generic Fetch] HTTP error {e.response.status_code} for: {e.request.url}"); return None
            except httpx.RequestError as e: logger.error(f"[Generic Fetch] Request error for {url}: {e}"); return None
            except Exception as e: logger.error(f"[Generic Fetch] Unexpected error fetching {url}: {e}", exc_info=True); return None

# =============================================================================
# Main SEC Risk Factor Scraper Class (API Focused)
# =============================================================================
class SECRiskFactorScraper:
    """Extracts risk factors from SEC 10-K filings using APIs."""

    def __init__(self, api_key=None, use_api_first=True):
        self.headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"}
        self.api_key = api_key
        self.use_api = api_key is not None # Simplified flag
        self.api_calls_remaining = MAX_API_CALLS if api_key else 0
        self.vectorizer = RiskVectorizer()
        self.async_processor = AsyncSECProcessor(api_key)
        logger.info(f"SECRiskFactorScraper initialized. API Enabled: {self.use_api}")
        if self.use_api: logger.info(f"Initial API calls available: {self.api_calls_remaining}")

    def track_api_call(self) -> bool:
        """Tracks API call usage. Returns True if call can proceed."""
        if not self.use_api: return False # No key, no calls
        if self.api_calls_remaining > 0:
            self.api_calls_remaining -= 1
            logger.info(f"API call used. Remaining: {self.api_calls_remaining}")
            return True
        else:
            logger.warning("API call limit reached.")
            return False

    # --- Kept for CIK lookup if needed by get_filing_urls_via_scraping ---
    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(4), retry=retry_if_exception_type(requests.exceptions.RequestException))
    def throttled_request(self, url: str) -> Optional[requests.Response]:
        """Synchronous throttled request with retry (used for scraping index search)."""
        # time.sleep(SEC_RATE_LIMIT_SLEEP) # Throttling now handled by AsyncLimiter for async calls
        try:
            response = requests.get(url, headers=self.headers, timeout=20.0)
            if response.status_code == 429: raise requests.exceptions.RequestException("Rate limited (429)")
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Throttled request failed for {url}: {e}. Retrying...")
            raise # Re-raise for tenacity

    def get_company_cik(self, ticker: str) -> Optional[str]:
        """Gets CIK for a ticker using SEC EDGAR browse."""
        ticker = ticker.upper()
        url = f"{SEC_BASE_URL}/cgi-bin/browse-edgar?CIK={ticker}&owner=exclude&action=getcompany"
        try:
            response = self.throttled_request(url)
            if not response: return None
            cik_match = re.search(r'CIK=(\d+)', response.text)
            if not cik_match: logger.error(f"Could not find CIK for {ticker}"); return None
            cik = cik_match.group(1).zfill(10)
            logger.info(f"Found CIK for {ticker}: {cik}")
            return cik
        except Exception as e:
            logger.error(f"Error getting CIK for {ticker}: {e}", exc_info=True)
            return None
    # --- End Kept Sync Methods ---

    def get_filing_urls_via_api(self, ticker: str, form_type: str = "10-K", year: Optional[int] = None) -> Optional[List[Dict]]:
        """Gets filing metadata using SEC Query API."""
        if not self.use_api or not SEC_QUERY_API_AVAILABLE or not QueryApi:
            logger.debug("Skipping QueryApi lookup (API disabled, lib unavailable, or no key).")
            return None
        if not self.track_api_call(): logger.warning("Skipping QueryApi lookup (limit reached)."); return None
        try:
            query_api = QueryApi(api_key=self.api_key)
            if year: query_str = f"ticker:{ticker} AND formType:\"{form_type}\" AND filedAt:[{year}-01-01 TO {year}-12-31]"
            else: query_str = f"ticker:{ticker} AND formType:\"{form_type}\""
            query = {"query": {"query_string": {"query": query_str}}, "from": "0", "size": "10", "sort": [{"filedAt": {"order": "desc"}}]}
            response = query_api.get_filings(query)
            filings = response.get('filings', [])
            if not filings: logger.info(f"No filings found via QueryApi for {ticker}, year {year}."); return None
            results = []
            for f in filings:
                acc_no = f.get('accessionNo'); date = f.get('filedAt','').split('T')[0]; yr = int(date.split('-')[0]) if date else None
                if acc_no and yr: results.append({'ticker': ticker, 'accession_no': acc_no, 'filing_date': date, 'filing_year': yr, 'form_type': f.get('formType'), 'cik': f.get('cik', '')})
            logger.info(f"Found {len(results)} filings via QueryApi for {ticker}, year {year}")
            return results
        except Exception as e: logger.error(f"QueryApi error for {ticker}, year {year}: {e}", exc_info=True); return None

    def get_filing_urls_via_scraping(self, ticker: str, form_type: str = "10-K", year: Optional[int] = None) -> List[Dict]:
        """Gets filing URLs by scraping SEC EDGAR search results (Sync)."""
        cik = self.get_company_cik(ticker)
        if not cik: return []
        if year: url = f"{SEC_BASE_URL}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form_type}&dateb={year}-12-31&datea={year}-01-01&owner=exclude&count=100"
        else: url = f"{SEC_BASE_URL}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form_type}&dateb=&owner=exclude&count=100"
        try:
            response = self.throttled_request(url) # Uses sync request
            if not response: return []
        except Exception as e:
            logger.error(f"Failed to get EDGAR search page {url}: {e}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'tableFile2'})
        if not table: logger.error(f"No filing table found for {ticker} at {url}"); return []
        results = []
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if len(cells) >= 4:
                form = cells[0].text.strip()
                if form != form_type: continue
                file_date_str = cells[3].text.strip()
                try:
                    filing_yr_found = int(file_date_str.split('-')[0])
                    if year and filing_yr_found != year: continue
                    detail_link_tag = cells[1].a
                    if not detail_link_tag or not detail_link_tag.has_attr('href'): continue
                    filing_detail_url = f"{SEC_BASE_URL}{detail_link_tag['href']}"
                    results.append({'ticker': ticker, 'filing_detail_url': filing_detail_url, 'filing_date': file_date_str, 'filing_year': filing_yr_found, 'form_type': form_type, 'cik': cik})
                except Exception as e: logger.error(f"Error parsing EDGAR search row for {ticker}: {e}")
        logger.info(f"Found {len(results)} filings via scraping EDGAR search for {ticker}, year {year}")
        return results

    # --- REMOVED: extract_risk_factors_via_scraping ---
    # --- REMOVED: _clean_xbrl ---

    def clean_risk_factor_text(self, text: str) -> str:
        """Basic cleaning for extracted risk factor text."""
        if not text: return ""
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        # Remove excessive newlines that might remain after HTML/text conversion
        text = re.sub(r'(\n\s*){3,}', '\n\n', text)
        # Simple removal of page numbers (might need refinement)
        # text = re.sub(r'\bPage \d+\b', '', text, flags=re.IGNORECASE)
        # text = re.sub(r'\s+[-–—]?\s*\d+\s*[-–—]?\s*$', '', text, flags=re.MULTILINE) # Page numbers at end of line
        text = re.sub(r'Table of Contents', '', text, flags=re.IGNORECASE)
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text

    def process_filing_to_dataframe(self, risk_text: str, ticker: str, filing_year: int, cik: Optional[str]) -> Optional[pd.DataFrame]:
        """Processes extracted text into a structured DataFrame with NLP enrichment."""
        try:
            if not risk_text: logger.warning(f"No risk text provided for {ticker} ({filing_year})"); return None
            text = self.clean_risk_factor_text(risk_text)
            if not text: logger.warning(f"Risk text became empty after cleaning for {ticker} ({filing_year})"); return None

            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50] # Filter short paragraphs early
            if not paragraphs: logger.warning(f"No substantial paragraphs found after cleaning/splitting for {ticker} ({filing_year})"); return None

            rows = []
            for para_idx, paragraph in enumerate(paragraphs):
                # Basic sentence splitting (consider using NLTK or SpaCy for complex cases)
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', paragraph) if len(s.strip()) > 10]
                for sent_idx, sentence in enumerate(sentences):
                    sentiment_score, _ = self.vectorizer.vectorize_sentence(sentence)
                    risk_keywords = self.vectorizer.extract_risk_keywords(sentence)
                    word_count = len(sentence.split())
                    rows.append({
                        "ticker": ticker, "year": filing_year, "cik": cik,
                        "paragraph_idx": para_idx, "sentence_idx": sent_idx,
                        "sentence": sentence, "word_count": word_count,
                        "contains_risk": "risk" in sentence.lower(),
                        "contains_uncertainty": any(term in sentence.lower() for term in ["could", "may", "might", "uncertain", "potential", "adverse"]),
                        "sentiment_score": sentiment_score,
                        "risk_keywords": risk_keywords
                    })

            if not rows: logger.warning(f"No sentences extracted for {ticker} ({filing_year})"); return None

            df = pd.DataFrame(rows) # Use pandas alias pd
            df = df.replace({np.nan: None}) # Handle potential NaNs before validation

            # Schema validation (using global schema)
            try:
                df = risk_factor_schema.validate(df, lazy=True) # Use lazy=True to collect all errors
            except pda.errors.SchemaErrors as err:
                 logger.warning(f"Pandera schema validation failed for {ticker} ({filing_year}):\n{err.failure_cases}")
                 # Optionally return None or raise error depending on strictness needed
                 # return None # Option: Fail processing if validation fails
            except Exception as e_val:
                 logger.error(f"Unexpected error during Pandera validation for {ticker} ({filing_year}): {e_val}")
                 # return None # Option: Fail processing

            logger.info(f"Processed {ticker} ({filing_year}): {len(df)} sentences.")
            return df
        except Exception as e_proc:
            logger.error(f"Error processing text to DataFrame for {ticker} ({filing_year}): {e_proc}", exc_info=True)
            return None

    def save_dataframe_as_parquet(self, df: pd.DataFrame, ticker: str, year: int) -> bool:
        """Saves DataFrame as Parquet (Sync function, run in thread)."""
        if df is None or df.empty: logger.warning(f"No data to save for {ticker} ({year})"); return False
        try:
            ticker_dir = PROCESSED_DATA_DIR / ticker; ticker_dir.mkdir(exist_ok=True)
            parquet_path = ticker_dir / f"{ticker}_{year}_risk.parquet"
            df_to_save = df.replace({np.nan: None}) # Ensure NaNs handled for pyarrow

            # Convert to Arrow Table with schema
            try:
                arrow_table = pa.Table.from_pandas(df_to_save, schema=arrow_schema, preserve_index=False)
            except Exception as e_arrow: # Catch broader errors during conversion
                logger.warning(f"PyArrow conversion with schema failed for {ticker} ({year}): {e_arrow}. Trying without schema.")
                try:
                    arrow_table = pa.Table.from_pandas(df_to_save, preserve_index=False) # Fallback to inferred schema
                except Exception as e_arrow_fallback:
                    logger.error(f"PyArrow conversion failed even without schema for {ticker} ({year}): {e_arrow_fallback}")
                    raise # Re-raise to trigger backup or fail

            pq.write_table(arrow_table, parquet_path, compression='ZSTD', coerce_timestamps='ms')
            logger.info(f"Saved Parquet data for {ticker} ({year}) to {parquet_path}")
            return True
        except Exception as e_save:
            logger.error(f"Error saving Parquet for {ticker} ({year}): {e_save}", exc_info=True)
            # Optional CSV backup (consider if needed)
            # try: ... df.to_csv(...) ... except ...
            return False

    async def process_company_years_async(self, ticker: str, years: List[int]):
        """Processes multiple years for a company concurrently."""
        ticker = ticker.upper()
        logger.info(f"Async processing {ticker} for years {years}")
        tasks = [self.process_company_filing_async(ticker, year) for year in years]
        results = await asyncio.gather(*tasks, return_exceptions=True) # Handle exceptions gracefully
        successful_count = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'success')
        logger.info(f"Completed async processing for {ticker}: {successful_count}/{len(years)} years successful.")
        # Process results to handle potential exceptions returned by gather
        final_results = []
        for i, res in enumerate(results):
             if isinstance(res, Exception):
                  logger.error(f"Task for {ticker} year {years[i]} resulted in unhandled exception: {res}", exc_info=res)
                  final_results.append({"ticker": ticker, "year": years[i], "status": "error", "error": f"Unhandled Exception: {type(res).__name__}"})
             else:
                  final_results.append(res) # Append dict result (success or handled failure/error)
        return {"ticker": ticker, "years_processed": len(years), "years_successful": successful_count, "results": final_results}

    async def process_company_filing_async(self, ticker: str, year: int) -> Dict:
        """
        Processes a single company filing (API-Focused).
        Flow: Find Index URL -> Find Doc URL -> Try Direct HTTP API -> Process/Save
        """
        # --- Initialize state variables ---
        risk_factors: Optional[str] = None
        filing_info: Dict = {}
        successful: bool = False
        cik: Optional[str] = None
        method_used: str = "unknown"
        filing_year: int = year
        accession_no: Optional[str] = None
        filing_detail_url: Optional[str] = None
        document_url: Optional[str] = None
        xbrl_compliant_flag: bool = False
        # --- End Initialization ---

        try:
            # === Step 1: Find Filing Info & Index Page URL ===
            query_api_results = None
            if self.use_api: # Try Query API first if enabled
                query_api_results = self.get_filing_urls_via_api(ticker, year=year) # Tracks call inside

            if query_api_results: # Found via API
                filing_info = query_api_results[0]
                accession_no = filing_info.get('accession_no')
                filing_year = filing_info.get('filing_year', year)
                cik = filing_info.get('cik')
                if accession_no and cik:
                    padded_cik = cik.lstrip("0")
                    acc_no_no_dash = accession_no.replace('-', '')
                    filing_detail_url = f"{SEC_ARCHIVES_URL}/edgar/data/{padded_cik}/{acc_no_no_dash}/{accession_no}-index.htm"
                    logger.info(f"Found filing via QueryApi (AccNo={accession_no}). Constructed Index URL.")
                else: logger.warning("QueryApi found filing but CIK/AccNo missing, cannot construct index URL.")
            else: # Try scraping index search if API failed or wasn't used
                logger.info(f"QueryApi failed or skipped. Attempting scraping EDGAR search for {ticker} ({year})")
                # Run sync scraping search in thread
                scraped_filings_info = await asyncio.to_thread(self.get_filing_urls_via_scraping, ticker, year=year)
                if scraped_filings_info:
                    filing_info = scraped_filings_info[0]
                    filing_year = filing_info.get('filing_year', filing_year)
                    if not cik: cik = filing_info.get('cik')
                    filing_detail_url = filing_info.get('filing_detail_url')
                    logger.info(f"Found filing via scraping index search: CIK={cik}, Year={filing_year}.")
                else:
                    logger.error(f"Could not find filing info via API or scraping for {ticker} ({year}).")
                    return {"ticker": ticker, "year": year, "cik": cik, "status": "failed", "error": "Filing info/index URL not found"}

            # === Step 2: Get Document URL from Index Page ===
            if not filing_detail_url:
                logger.error(f"Filing Index Detail URL is missing for {ticker} ({year}).")
                return {"ticker": ticker, "year": year, "cik": cik, "status": "failed", "error": "Missing filing_detail_url"}

            index_page_content = await self.async_processor.fetch_filing(filing_detail_url, self.headers)
            if not index_page_content:
                logger.error(f"Failed to fetch index page content: {filing_detail_url}")
                return {"ticker": ticker, "year": year, "cik": cik, "status": "failed", "error": "Failed to fetch index page"}

            # Parse index page to find the primary 10-K document URL
            soup = BeautifulSoup(index_page_content, 'html.parser')
            table = soup.find('table', {'class': 'tableFile'})
            if table:
                for row in table.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        doc_link_tag = cells[2].a; file_type = cells[3].text.strip()
                        if doc_link_tag and doc_link_tag.has_attr('href') and \
                           (doc_link_tag.text.lower().endswith('.htm') or doc_link_tag.text.lower().endswith('.html')) and \
                           file_type == '10-K':
                            doc_href = doc_link_tag['href']
                            document_url = f"{SEC_BASE_URL}{doc_href}" if doc_href.startswith('/') else f"{filing_detail_url.rsplit('/', 1)[0]}/{doc_href}"
                            logger.info(f"Found 10-K document link: {document_url}")
                            break
            if not document_url and filing_detail_url.endswith(".htm"): # Fallback for older filings
                 logger.warning(f"No 10-K link found, using index URL as document URL: {filing_detail_url}")
                 document_url = filing_detail_url

            # === Step 3: Attempt Extraction via Direct HTTP API ===
            if not document_url:
                 logger.error(f"Failed to find document URL for {ticker} ({year}).")
                 return {"ticker": ticker, "year": year, "cik": cik, "status": "failed", "error": "Document URL not found"}

            xbrl_compliant_flag = '/ix?doc=' in document_url # Basic check

            if self.use_api: # Only try API if enabled and key exists
                logger.info(f"Attempting Direct HTTP Extractor API for {ticker} ({year})")
                method_used = "api_http"
                if self.track_api_call():
                    api_result = await self.async_processor.fetch_section_via_extractor_http(document_url, item="1A")
                    if api_result and api_result.get('content'):
                        risk_factors = api_result.get('content')
                        # API returns cleaned text, no need for _clean_xbrl based on docs
                        successful = True
                        logger.info(f"Successfully extracted via Direct HTTP API for {ticker} ({filing_year})")
                    else:
                        logger.warning(f"Direct HTTP Extractor API failed or returned no content for {ticker} ({filing_year}).")
                else:
                    logger.warning("API call limit reached, cannot use Extractor API.")
            else:
                logger.info("API is disabled, skipping Extractor API call.")


            # === Step 4: Process and Save (if API succeeded) ===
            if successful and risk_factors:
                method_used = "api_http" # Confirm method if successful here
                # Save raw content (run sync I/O in thread)
                def save_raw_sync():
                    company_dir = RAW_DATA_DIR / ticker; company_dir.mkdir(exist_ok=True)
                    raw_path = company_dir / f"{ticker}_{filing_year}_10K_risk_factors_raw.txt"
                    try:
                        with open(raw_path, "w", encoding="utf-8") as f: f.write(risk_factors)
                        logger.info(f"Saved raw risk factors to {raw_path}"); return True
                    except Exception as e_raw: logger.error(f"Failed to save raw text: {e_raw}"); return False
                await asyncio.to_thread(save_raw_sync)

                # Process into DataFrame (Sync CPU - ok for now, consider thread if slow)
                df = self.process_filing_to_dataframe(risk_factors, ticker, filing_year, cik)

                if df is not None and not df.empty:
                    # Save Parquet (run sync I/O in thread)
                    save_successful = await asyncio.to_thread(self.save_dataframe_as_parquet, df, ticker, filing_year)
                    if save_successful:
                        return {"ticker": ticker, "year": filing_year, "cik": cik, "status": "success", "method": method_used, "sentences": len(df), "xbrl_compliant": xbrl_compliant_flag, "dataframe": df}
                    else: # Treat save failure as error
                        logger.error(f"Failed to save Parquet data for {ticker} ({filing_year}).")
                        return {"ticker": ticker, "year": filing_year, "cik": cik, "status": "error", "error": "Failed to save Parquet data"}
                else: # DataFrame processing failed
                     logger.warning(f"Processing resulted in empty/None DataFrame for {ticker} ({filing_year}).")
                     # Keep success=True but return a failure status for this step
                     return {"ticker": ticker, "year": filing_year, "cik": cik, "status": "failed", "error": "DataFrame processing failed", "method": method_used}

            # === Step 5: Handle Failure (if API failed/skipped) ===
            else: # Not successful via API
                 logger.warning(f"Failed to extract risk factors for {ticker} ({year}) using API. Scraping fallback removed.")
                 return {"ticker": ticker, "year": year, "cik": cik, "status": "failed", "error": "API extraction failed/skipped", "method": method_used}

        # === General Exception Handling ===
        except Exception as e:
            logger.exception(f"Unhandled error processing {ticker} ({year}): {e}", exc_info=True)
            return {"ticker": ticker, "year": year, "cik": cik if 'cik' in locals() and cik is not None else None, "status": "error", "error": f"Unhandled Exception: {type(e).__name__}: {str(e)}"}

    # =========================================================================
    # Analysis and Main Execution (Keep these, they rely on successful runs)
    # =========================================================================
    def process_company(self, ticker: str, start_year: int = 2018, end_year: Optional[int] = None):
        """Processes filings for a company, aggregates, analyzes."""
        if end_year is None: end_year = datetime.now().year
        ticker = ticker.upper()
        logger.info(f"Processing {ticker} from {start_year} to {end_year}")
        years = list(range(start_year, end_year + 1))
        try: loop = asyncio.get_running_loop()
        except RuntimeError: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        company_results = loop.run_until_complete(self.process_company_years_async(ticker, years))

        successful_results = [r for r in company_results.get('results', []) if isinstance(r, dict) and r.get('status') == 'success']
        dataframes = [r.get('dataframe') for r in successful_results if r.get('dataframe') is not None and not r['dataframe'].empty]

        analysis_performed = False
        if dataframes:
            try:
                self.analyze_risk_factors(ticker, dataframes)
                analysis_performed = True
            except Exception as e_analyze:
                 logger.error(f"Analysis failed for {ticker}: {e_analyze}", exc_info=True)

        summary = {
            "ticker": ticker, "years_requested": len(years),
            "years_processed_successfully": len(successful_results),
            "years_with_data_for_analysis": len(dataframes),
            "analysis_performed": analysis_performed,
            "years_covered_success": sorted([r.get('year') for r in successful_results]),
            "total_sentences_success": sum(r.get('sentences', 0) for r in successful_results),
            "api_calls_used": MAX_API_CALLS - self.api_calls_remaining if self.use_api else 0,
            "status": "success" if successful_results else "failed",
            "methods_used": {r.get('year'): r.get('method') for r in successful_results}, # Track method per year
            "last_updated": datetime.now().isoformat()
        }
        summary_path = PROCESSED_DATA_DIR / f"{ticker}_summary.json"
        try:
            with open(summary_path, 'w') as f: json.dump(summary, f, indent=4)
        except Exception as e_json: logger.error(f"Failed to save summary JSON for {ticker}: {e_json}")
        logger.info(f"Completed processing for {ticker}: {summary['years_processed_successfully']}/{summary['years_requested']} years successful.")
        return summary

    def analyze_risk_factors(self, ticker: str, dataframes: List[pd.DataFrame]):
        """Analyzes combined risk factor data across years."""
        if not dataframes: logger.warning(f"No dataframes provided for analysis for {ticker}"); return None
        try:
            combined_df = pd.concat(dataframes, ignore_index=True)
            if combined_df.empty: logger.warning(f"Combined dataframe is empty for {ticker}"); return None
            years = sorted(combined_df['year'].unique())
            stats = []
            for year in years:
                year_df = combined_df[combined_df['year'] == year]
                if year_df.empty: continue
                cik = year_df['cik'].iloc[0] if not year_df['cik'].isna().all() else None
                all_keywords = [kw for sublist in year_df['risk_keywords'] if sublist for kw in sublist] # Flatten list
                keyword_counts = pd.Series(all_keywords).value_counts().to_dict()
                top_keywords = dict(sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True)[:10])
                year_stats = {
                    'ticker': ticker, 'year': int(year), 'cik': cik, # Ensure year is int
                    'total_sentences': len(year_df),
                    'total_paragraphs': int(year_df['paragraph_idx'].nunique()), # Ensure int
                    'avg_sentence_length': round(year_df['word_count'].mean(), 2) if not year_df['word_count'].empty else 0.0,
                    'risk_sentence_count': int(year_df['contains_risk'].sum()),
                    'risk_sentence_pct': round((year_df['contains_risk'].mean()) * 100, 2) if not year_df.empty else 0.0,
                    'uncertainty_sentence_count': int(year_df['contains_uncertainty'].sum()),
                    'uncertainty_sentence_pct': round((year_df['contains_uncertainty'].mean()) * 100, 2) if not year_df.empty else 0.0,
                    'avg_sentiment_score': round(year_df['sentiment_score'].mean(), 4) if not year_df['sentiment_score'].isna().all() else 0.0,
                    'top_risk_keywords': top_keywords
                }
                stats.append(year_stats)

            if not stats: logger.warning(f"No yearly stats generated for {ticker}"); return None
            stats_df = pd.DataFrame(stats)

            # Calculate year-over-year changes
            if len(years) > 1:
                stats_df = stats_df.sort_values('year')
                for col in ['total_sentences', 'total_paragraphs', 'avg_sentence_length', 'risk_sentence_pct', 'uncertainty_sentence_pct', 'avg_sentiment_score', 'risk_sentence_count', 'uncertainty_sentence_count']:
                    stats_df[f'delta_{col}'] = stats_df[col].diff().round(4)
                    # Calculate pct change, handling potential division by zero or NaN/inf results
                    pct_change = stats_df[col].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100
                    stats_df[f'pct_change_{col}'] = pct_change.round(2)

            # Save analysis results
            analysis_dir = ANALYSIS_DIR / ticker; analysis_dir.mkdir(exist_ok=True)
            analysis_csv_path = analysis_dir / f"{ticker}_risk_analysis.csv"
            analysis_parquet_path = analysis_dir / f"{ticker}_risk_analysis.parquet"
            keywords_path = analysis_dir / f"{ticker}_top_keywords.json"

            # Prepare for saving (handle complex types)
            stats_df_for_csv = stats_df.copy()
            # Convert list/dict columns to JSON strings for CSV/Parquet compatibility if direct save fails
            stats_df_for_csv['top_risk_keywords'] = stats_df_for_csv['top_risk_keywords'].apply(json.dumps)
            stats_df_for_csv.to_csv(analysis_csv_path, index=False)
            logger.info(f"Saved analysis CSV to {analysis_csv_path}")

            # Try saving Parquet without complex columns first
            try:
                pq.write_table(pa.Table.from_pandas(stats_df.drop(columns=['top_risk_keywords']), preserve_index=False), analysis_parquet_path, compression='ZSTD')
                logger.info(f"Saved analysis Parquet to {analysis_parquet_path}")
            except Exception as e_pq:
                 logger.error(f"Failed to save analysis Parquet for {ticker}: {e_pq}. Check complex column handling.")

            # Save keywords JSON
            keywords_data = {int(row['year']): row['top_risk_keywords'] for index, row in stats_df.iterrows()}
            with open(keywords_path, 'w') as f: json.dump(keywords_data, f, indent=4)
            logger.info(f"Saved top keywords JSON to {keywords_path}")

            # Create visualization (optional)
            self.create_analysis_plot(ticker, stats_df, analysis_dir)

            logger.info(f"Completed risk factor analysis for {ticker}")
            return stats_df

        except Exception as e: logger.error(f"Error analyzing risk factors for {ticker}: {e}", exc_info=True); return None

    def create_analysis_plot(self, ticker: str, stats_df: pd.DataFrame, analysis_dir: Path):
        """Creates and saves the analysis plot using a non-interactive backend."""
        try:
            # --- FIX: Set backend BEFORE importing pyplot ---
            import matplotlib
            # Use 'Agg' backend suitable for non-GUI environments/threads
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            # --- End FIX ---

            # Check if stats_df is empty or None before proceeding
            if stats_df is None or stats_df.empty:
                logger.warning(f"Stats DataFrame is empty for {ticker}. Skipping plot generation.")
                return

            plt.style.use('ggplot')
            fig, axes = plt.subplots(2, 2, figsize=(14, 11)) # Create figure and axes
            axes = axes.flatten() # Flatten axes array for easy indexing

            # --- Plotting Logic (remains the same) ---

            # Plot 1: Risk/Uncertainty Pct
            axes[0].plot(stats_df['year'], stats_df['risk_sentence_pct'], marker='o', label='% Risk Sentences', linewidth=2)
            axes[0].plot(stats_df['year'], stats_df['uncertainty_sentence_pct'], marker='s', label='% Uncertainty Sentences', linewidth=2)
            axes[0].set_title('Risk & Uncertainty Mention (%)')
            axes[0].set_ylabel('% of Sentences')

            # Plot 2: Document Size (using twin axes)
            ax2b = axes[1].twinx()
            axes[1].bar(stats_df['year'] - 0.2, stats_df['total_sentences'], width=0.4, alpha=0.7, label='Sentences', color='tab:blue')
            ax2b.bar(stats_df['year'] + 0.2, stats_df['total_paragraphs'], width=0.4, alpha=0.6, label='Paragraphs', color='tab:orange')
            axes[1].set_ylabel('Total Sentences', color='tab:blue'); axes[1].tick_params(axis='y', labelcolor='tab:blue')
            ax2b.set_ylabel('Total Paragraphs', color='tab:orange'); ax2b.tick_params(axis='y', labelcolor='tab:orange')
            axes[1].set_title('Document Size (Sentences/Paragraphs)')
            lines, labels = axes[1].get_legend_handles_labels(); lines2, labels2 = ax2b.get_legend_handles_labels()
            ax2b.legend(lines + lines2, labels + labels2, loc='upper left') # Combined legend

            # Plot 3: Avg Sentiment
            axes[2].plot(stats_df['year'], stats_df['avg_sentiment_score'], marker='d', color='green', linewidth=2, label='Avg. Sentiment')
            axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[2].set_title('Average Sentiment Score')
            axes[2].set_ylabel('Score (-1 to 1)')

            # Plot 4: YoY Change (Example: Sentence Count %)
            if 'pct_change_total_sentences' in stats_df.columns and len(stats_df) > 1:
                 # Plot only from the second year onwards where change is defined
                 plot_years = stats_df['year'][1:]
                 plot_values = stats_df['pct_change_total_sentences'].iloc[1:].fillna(0)
                 axes[3].bar(plot_years, plot_values, alpha=0.7, label='% Change Sentences')
                 axes[3].set_title('YoY Change (%) in Total Sentences')
                 axes[3].set_ylabel('% Change')
                 axes[3].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            else:
                 axes[3].text(0.5, 0.5, 'YoY Change requires >1 year', horizontalalignment='center', verticalalignment='center', transform=axes[3].transAxes)
                 axes[3].set_title('YoY Change (%)')

            # Common formatting for all axes
            all_years = stats_df['year'].unique()
            for ax in axes:
                ax.set_xlabel('Year')
                ax.grid(True, linestyle='--', alpha=0.6)
                # Set x-ticks only if there are years to plot
                if len(all_years) > 0:
                    ax.set_xticks(all_years)
                    ax.tick_params(axis='x', rotation=45)
                # Add legend if it has labeled elements (except for plot 2 handled by combined legend)
                handles, labels = ax.get_legend_handles_labels()
                if labels and ax != axes[1]:
                    ax.legend()

            fig.suptitle(f'{ticker} - Risk Factor Analysis ({stats_df["year"].min()}-{stats_df["year"].max()})', fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout for suptitle

            # Save the plot
            plt_path = analysis_dir / f"{ticker}_risk_analysis_plot.png"
            plt.savefig(plt_path, dpi=300, bbox_inches='tight')
            plt.close(fig) # IMPORTANT: Close the figure to release memory
            logger.info(f"Saved analysis plot to {plt_path}")

        except ImportError:
            logger.warning("Matplotlib not installed. Skipping plot generation.")
        except Exception as e_plot:
            logger.error(f"Error creating plot for {ticker}: {e_plot}", exc_info=True)

# =============================================================================
# Main Execution Logic
# =============================================================================
async def process_multiple_companies(tickers: List[str], start_year: int, end_year: Optional[int] = None, max_concurrent: int = 5):
    """Processes multiple companies concurrently."""
    if end_year is None: end_year = datetime.now().year
    scraper = SECRiskFactorScraper(api_key=SEC_API_KEY, use_api_first=True) # API is primary now
    all_results = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(ticker):
        async with semaphore:
            logger.info(f"Starting processing for: {ticker}")
            try:
                 # Use process_company which handles years internally now
                 result = await asyncio.to_thread(scraper.process_company, ticker, start_year, end_year)
                 # result = await scraper.process_company_years_async(ticker, list(range(start_year, end_year + 1))) # Alt: if keeping years separate
                 logger.info(f"Finished processing for: {ticker}")
                 return ticker, result
            except Exception as e_proc:
                 logger.error(f"Error in process_with_semaphore for {ticker}: {e_proc}", exc_info=True)
                 return ticker, {"ticker": ticker, "status": "error", "error": f"Outer processing error: {e_proc}"}

    tasks = [process_with_semaphore(ticker.upper()) for ticker in tickers]
    # Use tqdm for progress if processing many tickers
    # for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Companies"):
    #      ticker, result = await future
    #      all_results[ticker] = result
    # Non-tqdm version:
    completed_tasks = await asyncio.gather(*tasks)
    for ticker, result in completed_tasks:
         all_results[ticker] = result

    if scraper.use_api: logger.info(f"Final API calls remaining: {scraper.api_calls_remaining}")
    return all_results

async def main():
    """Main async execution function"""
    companies = [
        # {"ticker": "AAPL", "name": "Apple Inc."},
        # {"ticker": "MSFT", "name": "Microsoft Corporation"},
        # {"ticker": "GOOGL", "name": "Alphabet Inc."},
        {"ticker": "AMZN", "name": "Amazon.com, Inc."},
        # {"ticker": "META", "name": "Meta Platforms, Inc."}
    ]
    tickers = [c["ticker"] for c in companies]
    start_year = 2022; end_year = 2023; max_concurrent = 3 # Adjust as needed

    logger.info(f"Starting SEC extraction for tickers: {tickers}, years: {start_year}-{end_year}")
    results = await process_multiple_companies(tickers, start_year, end_year, max_concurrent)

    # --- Save overall results summary ---
    results_path = PROCESSED_DATA_DIR / f"batch_results_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # Basic cleaning for JSON (remove DataFrames from detailed results if they exist)
    clean_summary_results = {}
    for ticker, summary_data in results.items():
        clean_summary = summary_data.copy() # Copy top-level summary
        # If detailed results were nested (they aren't in process_company output, but were in process_years)
        if 'results' in clean_summary and isinstance(clean_summary['results'], list):
             clean_summary['results'] = [{k: v for k, v in r.items() if k != 'dataframe'} for r in clean_summary['results'] if isinstance(r, dict)]
        clean_summary_results[ticker] = clean_summary

    try:
        with open(results_path, 'w') as f: json.dump(clean_summary_results, f, indent=4)
        logger.info(f"Overall batch summary saved to {results_path}")
    except Exception as e_json: logger.error(f"Failed to save overall results summary JSON: {e_json}")
    # --- End Save Summary ---

    # --- Print final summary stats ---
    if results: # Check if results dict is not empty
         total_companies = len(results)
         successful_companies = sum(1 for r in results.values() if isinstance(r, dict) and r.get('status') == 'success' and r.get('years_processed_successfully', 0) > 0)
         total_years_attempted = sum(r.get('years_requested', 0) for r in results.values() if isinstance(r, dict))
         total_years_successful = sum(r.get('years_processed_successfully', 0) for r in results.values() if isinstance(r, dict))
         company_success_rate = (successful_companies / total_companies * 100) if total_companies else 0
         year_success_rate = (total_years_successful / total_years_attempted * 100) if total_years_attempted else 0
         logger.info(f"--- Final Batch Summary ---")
         logger.info(f"Companies processed: {total_companies}")
         logger.info(f"Successful companies (at least 1 year): {successful_companies} ({company_success_rate:.1f}%)")
         logger.info(f"Total years attempted: {total_years_attempted}")
         logger.info(f"Total years successful: {total_years_successful} ({year_success_rate:.1f}%)")
    # --- End Print Stats ---

if __name__ == "__main__":
    try:
        try: loop = asyncio.get_running_loop()
        except RuntimeError: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
    except KeyboardInterrupt: logger.info("Script interrupted by user.")
    finally: logger.info("Script finished.")