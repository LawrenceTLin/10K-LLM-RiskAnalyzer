#!/usr/bin/env python3
"""
Enhanced SEC 10-K Risk Factor Scraper

A production-grade hybrid scraper for extracting risk factors from 
SEC 10-K filings with enterprise features: async processing, 
schema validation, XBRL handling, and efficient storage.
"""

import os
import re
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from bs4 import BeautifulSoup

# Enhanced dependencies
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from unstructured.partition.html import partition_html
from tqdm import tqdm
from dotenv import load_dotenv
import asyncio
import httpx
from aiolimiter import AsyncLimiter

# Data validation and storage
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import schema as pa_schema, Table
import pandera as pd
from pandera import Column, DataFrameSchema, Check

# Optional NLP enhancements
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Load environment variables
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("sec_scraper.log")
    ]
)
logger = logging.getLogger(__name__)

# Setup directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "sec_filings"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed" / "sec_filings"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis" / "risk_factors"

# Create necessary directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, ANALYSIS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Constants
SEC_BASE_URL = "https://www.sec.gov"
SEC_ARCHIVES_URL = f"{SEC_BASE_URL}/Archives"
SEC_RATE_LIMIT_SLEEP = 0.1  # Sleep 100ms between requests
USER_AGENT = "RiskFactorAnalysisResearch research@example.edu"

# Track API usage
API_CALL_COUNT = 0
MAX_API_CALLS = 100  # Free tier limit

# Define Pandera schema for validation
risk_factor_schema = DataFrameSchema({
    "ticker": Column(str, Check.str_length(min_value=1, max_value=5)),
    "year": Column(int, Check.in_range(1993, datetime.now().year)),
    "cik": Column(str, nullable=True),
    "paragraph_idx": Column(int),
    "sentence_idx": Column(int),
    "sentence": Column(str, Check.str_length(min_value=10)),
    "word_count": Column(int, Check.greater_than(0)),
    "contains_risk": Column(bool),
    "contains_uncertainty": Column(bool),
    "sentiment_score": Column(float, Check.in_range(-1, 1), nullable=True),
    "risk_keywords": Column(list, nullable=True)
})

# Define PyArrow schema for Parquet
arrow_schema = pa_schema([
    ("ticker", pa.string()),
    ("year", pa.int16()),
    ("cik", pa.string()),
    ("paragraph_idx", pa.int32()),
    ("sentence_idx", pa.int32()),
    ("sentence", pa.string()),
    "word_count", pa.int32(),
    ("contains_risk", pa.bool_()),
    ("contains_uncertainty", pa.bool_()),
    ("sentiment_score", pa.float32()),
    ("risk_keywords", pa.list_(pa.string()))
])


class RiskVectorizer:
    """Process text with NLP models to extract financial sentiment"""
    
    def __init__(self, model_name="ProsusAI/finbert"):
        """Initialize NLP models"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available. Sentiment analysis disabled.")
            self.model = None
            self.tokenizer = None
            return
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info(f"Loaded {model_name} for sentiment analysis")
        except Exception as e:
            logger.error(f"Failed to load NLP models: {e}")
            self.model = None
            self.tokenizer = None
    
    def vectorize_sentence(self, text):
        """
        Extract sentiment scores from text using FinBERT
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentiment_score, sentiment_label)
        """
        if not self.model or not self.tokenizer:
            return 0.0, "neutral"
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            probs = outputs.logits.softmax(dim=1).detach().numpy()[0]  # [negative, neutral, positive]
            
            # Convert to a single score from -1 to 1
            # -1 (negative), 0 (neutral), 1 (positive)
            score = probs[2] - probs[0]  # positive - negative
            
            # Determine label
            if probs[0] > probs[1] and probs[0] > probs[2]:
                label = "negative"
            elif probs[2] > probs[1] and probs[2] > probs[0]:
                label = "positive"
            else:
                label = "neutral"
                
            return score, label
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.0, "neutral"
    
    def extract_risk_keywords(self, text):
        """
        Extract risk-related keywords from text
        
        Args:
            text: Input text
            
        Returns:
            List of risk keywords
        """
        risk_terms = [
            "risk", "uncertainty", "adverse", "negative", "volatility", 
            "decline", "failure", "litigation", "regulatory", "competitive",
            "liability", "disruption", "fluctuation", "economic", "pandemic",
            "recession", "inflationary", "cybersecurity", "breach", "compliance",
            "delay", "shortage", "downturn", "debt", "lawsuit"
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in risk_terms:
            if term in text_lower:
                found_terms.append(term)
                
        return found_terms


class AsyncSECProcessor:
    """Asynchronous processor for SEC filings"""
    
    def __init__(self, api_key=None):
        """Initialize async processor with rate limiting"""
        self.api_key = api_key
        self.limiter = AsyncLimiter(10, 1)  # SEC rate limits: 10 requests per second
        logger.info("Initialized AsyncSECProcessor with rate limiting")
    
    async def fetch_filing(self, url, headers=None):
        """
        Fetch a filing with async request and rate limiting
        
        Args:
            url: URL to fetch
            headers: Request headers
            
        Returns:
            Response text
        """
        if headers is None:
            headers = {
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
                "Host": "www.sec.gov"
            }
        
        async with self.limiter:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, headers=headers, timeout=30)
                    
                if response.status_code == 200:
                    return response.text
                else:
                    logger.error(f"Failed async request: {url}, Status: {response.status_code}")
                    return None
            except Exception as e:
                logger.error(f"Async request error for {url}: {e}")
                return None
    
    async def process_filing_batch(self, filing_urls, headers=None):
        """
        Process a batch of filings concurrently
        
        Args:
            filing_urls: List of URLs to process
            headers: Request headers
            
        Returns:
            List of response texts
        """
        tasks = [self.fetch_filing(url, headers) for url in filing_urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_api_batch(self, accession_numbers):
        """
        Process a batch of filings via API concurrently
        
        Args:
            accession_numbers: List of accession numbers
            
        Returns:
            List of API responses
        """
        if not self.api_key:
            return [None] * len(accession_numbers)
            
        async with self.limiter:
            tasks = []
            for acc_no in accession_numbers:
                tasks.append(self.fetch_api_filing(acc_no))
                
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def fetch_api_filing(self, accession_no):
        """
        Fetch a filing via SEC API
        
        Args:
            accession_no: Accession number
            
        Returns:
            API response
        """
        if not self.api_key:
            return None
            
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.sec-api.io/extractor/{accession_no}?section=1A",
                    headers=headers,
                    timeout=30
                )
                
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error for {accession_no}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"API request error for {accession_no}: {e}")
            return None


class SECRiskFactorScraper:
    """Enhanced scraper for extracting risk factors from SEC 10-K filings"""
    
    def __init__(self, api_key=None, use_api_first=True):
        """Initialize scraper with API settings and NLP tools"""
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        }
        self.api_key = api_key
        self.use_api_first = use_api_first and api_key is not None
        self.api_calls_remaining = MAX_API_CALLS if api_key else 0
        
        # Initialize NLP tools
        self.vectorizer = RiskVectorizer()
        
        # Initialize async processor
        self.async_processor = AsyncSECProcessor(api_key)
        
        logger.info(f"SEC Risk Factor Scraper initialized. API first: {self.use_api_first}")
        if self.use_api_first:
            logger.info(f"API calls available: {self.api_calls_remaining}")
    
    def track_api_call(self):
        """Track API usage and check if we're out of calls"""
        if self.api_calls_remaining > 0:
            self.api_calls_remaining -= 1
            logger.info(f"API call used. Remaining: {self.api_calls_remaining}")
            return True
        return False
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=60),
           stop=stop_after_attempt(5),
           retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)))
    def throttled_request(self, url):
        """Make a request to SEC with proper throttling and retry logic"""
        time.sleep(SEC_RATE_LIMIT_SLEEP)  # Respect SEC rate limit
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                # Too many requests, raise exception for retry
                logger.warning(f"Rate limited by SEC (429). Retrying with backoff...")
                raise requests.exceptions.RequestException("Rate limited")
            else:
                logger.error(f"Failed request: {url}, Status: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Request error for {url}: {e}")
            raise
    
    def get_company_cik(self, ticker):
        """
        Get the CIK (Central Index Key) for a company ticker
        
        Args:
            ticker (str): Company ticker symbol
            
        Returns:
            str: CIK number with leading zeros
        """
        ticker = ticker.upper()
        url = f"{SEC_BASE_URL}/cgi-bin/browse-edgar?CIK={ticker}&owner=exclude&action=getcompany"
        
        response = self.throttled_request(url)
        if not response:
            return None
        
        # Extract CIK from the response
        cik_match = re.search(r'CIK=(\d+)', response.text)
        if not cik_match:
            logger.error(f"Could not find CIK for {ticker}")
            return None
        
        # SEC expects 10 digits with leading zeros
        cik = cik_match.group(1).zfill(10)
        logger.info(f"Found CIK for {ticker}: {cik}")
        return cik
    
    def get_filing_urls_via_api(self, ticker, form_type="10-K", year=None):
        """Get filing URLs using SEC API if available"""
        if not self.api_key or not self.track_api_call():
            return None
            
        try:
            from sec_api import QueryApi
            query_api = QueryApi(api_key=self.api_key)
            
            # Build query for specific year if provided
            if year:
                query = {
                    "query": {
                        "query_string": {
                            "query": f"ticker:{ticker} AND formType:\"{form_type}\" AND filedAt:[{year}-01-01 TO {year}-12-31]"
                        }
                    },
                    "from": "0",
                    "size": "10",
                    "sort": [{"filedAt": {"order": "desc"}}]
                }
            else:
                query = {
                    "query": {
                        "query_string": {
                            "query": f"ticker:{ticker} AND formType:\"{form_type}\""
                        }
                    },
                    "from": "0",
                    "size": "10",
                    "sort": [{"filedAt": {"order": "desc"}}]
                }
            
            # Execute query
            response = query_api.get_filings(query)
            filings = response.get('filings', [])
            
            if not filings:
                logger.warning(f"No filings found via API for {ticker}")
                return None
                
            # Extract filing URLs
            results = []
            for filing in filings:
                accession_no = filing.get('accessionNo')
                filing_date = filing.get('filedAt', '').split('T')[0] if filing.get('filedAt') else None
                filing_year = int(filing_date.split('-')[0]) if filing_date else None
                
                # Store filing info with CIK
                if accession_no and filing_year:
                    results.append({
                        'ticker': ticker,
                        'accession_no': accession_no,
                        'filing_date': filing_date,
                        'filing_year': filing_year,
                        'form_type': form_type,
                        'cik': filing.get('cik', '')
                    })
            
            logger.info(f"Found {len(results)} filings via API for {ticker}")
            return results
            
        except Exception as e:
            logger.error(f"API error for {ticker}: {e}")
            return None
    
    def get_filing_urls_via_scraping(self, ticker, form_type="10-K", year=None):
        """Get filing URLs by scraping SEC EDGAR"""
        cik = self.get_company_cik(ticker)
        if not cik:
            logger.error(f"Could not find CIK for {ticker}")
            return []
            
        # Build search URL
        if year:
            url = f"{SEC_BASE_URL}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form_type}&dateb={year}-12-31&datea={year}-01-01&owner=exclude&count=100"
        else:
            url = f"{SEC_BASE_URL}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form_type}&dateb=&owner=exclude&count=100"
        
        response = self.throttled_request(url)
        if not response:
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'tableFile2'})
        
        if not table:
            logger.error(f"No filing table found for {ticker}")
            return []
            
        results = []
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = row.find_all('td')
            if len(cells) >= 4:
                form = cells[0].text.strip()
                if form != form_type:
                    continue
                    
                filing_date = cells[3].text.strip()
                try:
                    filing_year = int(filing_date.split('-')[0])
                    
                    # Skip if not the requested year
                    if year and filing_year != year:
                        continue
                        
                    # Get the filing detail URL
                    filing_detail_url = cells[1].a['href']
                    full_detail_url = f"{SEC_BASE_URL}{filing_detail_url}"
                    
                    # Store filing info
                    results.append({
                        'ticker': ticker,
                        'filing_detail_url': full_detail_url,
                        'filing_date': filing_date,
                        'filing_year': filing_year,
                        'form_type': form_type,
                        'cik': cik  # Include CIK
                    })
                except Exception as e:
                    logger.error(f"Error parsing row for {ticker}: {e}")
        
        logger.info(f"Found {len(results)} filings via scraping for {ticker}")
        return results
    
    def _clean_xbrl(self, text):
        """Remove XBRL namespace declarations and tags"""
        # Remove namespace declarations
        text = re.sub(r'\bxmlns:\w+="[^"]+"', '', text)
        
        # Remove ix: tags but keep content
        text = re.sub(r'<ix:[^>]*>(.*?)</ix:[^>]*>', r'\1', text)
        
        return text
    
    def extract_risk_factors_via_api(self, ticker, accession_no, filing_year):
        """Extract risk factors using SEC API if available"""
        if not self.api_key or not self.track_api_call():
            return None
            
        try:
            from sec_api import ExtractorApi
            extractor_api = ExtractorApi(api_key=self.api_key)
            
            # Extract Item 1A (Risk Factors)
            risk_section = extractor_api.get_section(accession_no, "1A")
            
            if not risk_section:
                logger.warning(f"No risk factors found via API for {ticker} ({filing_year})")
                return None
                
            content = risk_section.get('content', '')
            if not content:
                logger.warning(f"Empty risk factor content via API for {ticker} ({filing_year})")
                return None
            
            # Check if content contains XBRL
            result = {
                "content": content,
                "xbrl_compliant": False
            }
            
            if re.search(r'<ix:header>', content):
                result["xbrl_compliant"] = True
                result["content"] = self._clean_xbrl(content)
                logger.info(f"Cleaned XBRL content for {ticker} ({filing_year})")
                
            logger.info(f"Successfully extracted risk factors via API for {ticker} ({filing_year})")
            return result
            
        except Exception as e:
            logger.error(f"API extraction error for {ticker} ({filing_year}): {e}")
            return None
    
    def extract_risk_factors_via_scraping(self, document_url, ticker, filing_year, filing_info):
        """Extract risk factors by scraping document"""
        response = self.throttled_request(document_url)
        if not response:
            return None
        
        content = response.text
        soup = BeautifulSoup(content, 'html.parser')
        
        # Check for XBRL compliance
        xbrl_compliant = bool(re.search(r'<ix:header>', content))
        if xbrl_compliant:
            content = self._clean_xbrl(content)
            soup = BeautifulSoup(content, 'html.parser')
            logger.info(f"Detected and cleaned XBRL for {ticker} ({filing_year})")
        
        # Try different methods to locate the Risk Factors section
        risk_factor_text = ""
        
        # Method 1: Look for item headings with pattern matching
        item_patterns = [
            r'item\s*1a\.?\s*risk\s*factors',
            r'item\s*1a[\.\)]\s*risk\s*factors',
            r'item\s*1a\.?\s*risks',
            r'risk\s*factors'
        ]
        
        # Find section based on headers or strong text
        for pattern in item_patterns:
            # Look for headings with this pattern
            for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b']):
                if tag.text and re.search(pattern, tag.text.lower(), re.IGNORECASE):
                    # Found a risk factors heading
                    logger.info(f"Found Risk Factors section heading for {ticker} ({filing_year})")
                    
                    # Try to extract text until the next item
                    section_text = []
                    next_item_found = False
                    
                    # Start from the next sibling of the heading's parent 
                    parent = tag.parent
                    current = parent.next_sibling
                    
                    # First try to collect text up to next section
                    while current and not next_item_found:
                        # Check if this is the next item (1B, 2, etc.)
                        next_item_patterns = [r'item\s*1b', r'item\s*2']
                        
                        # Check in current element and its children for next section
                        current_text = current.get_text().lower() if hasattr(current, 'get_text') else ""
                        if any(re.search(p, current_text, re.IGNORECASE) for p in next_item_patterns):
                            next_item_found = True
                        
                        if not next_item_found and hasattr(current, 'name'):
                            # Add text content if it's a paragraph, div, or similar
                            if current.name in ['p', 'div', 'span', 'text']:
                                section_text.append(current.get_text().strip())
                        current = current.next_sibling
                    
                    if section_text:
                        risk_factor_text = "\n\n".join(section_text)
                        break
            
            if risk_factor_text:
                break
        
        # If all methods failed, use unstructured.io to extract whole document
        if not risk_factor_text:
            try:
                elements = partition_html(text=content)
                full_text = "\n".join([e.text for e in elements if hasattr(e, 'text')])
                
                # Try to find risk section in structured text
                for pattern in item_patterns:
                    match = re.search(pattern, full_text, re.IGNORECASE)
                    if match:
                        start_pos = match.end()
                        
                        # Find the end (next item)
                        next_item_match = re.search(r'item\s*1b|item\s*2', full_text[start_pos:], re.IGNORECASE)
                        
                        if next_item_match:
                            end_pos = start_pos + next_item_match.start()
                            risk_factor_text = full_text[start_pos:end_pos].strip()
                        else:
                            # If no next item found, limit to a reasonable size
                            risk_factor_text = full_text[start_pos:start_pos + 100000].strip()
                        
                        logger.info(f"Found Risk Factors using unstructured for {ticker} ({filing_year})")
                        break
            except Exception as e:
                logger.error(f"Unstructured.io error for {ticker} ({filing_year}): {e}")
        
        if risk_factor_text:
            return {
                "content": risk_factor_text,
                "xbrl_compliant": xbrl_compliant,
                "cik": filing_info.get('cik', '')  # Include CIK
            }
        
        return None
    
    def clean_risk_factor_text(self, text):
        """Clean extracted risk factor text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers often found in SEC documents
        text = re.sub(r'\s*\d+\s*', ' ', text)
        
        # Remove common SEC filing artifacts
        text = re.sub(r'Table of Contents', '', text)
        
        # Replace weird unicode characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Split into paragraphs and rejoin (helps with formatting)
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        return '\n\n'.join(paragraphs)
    
    def process_filing_to_dataframe(self, risk_text, ticker, filing_year, cik=None):
        """Process risk factor text into structured DataFrame with NLP enrichment"""
        try:
            if not risk_text:
                logger.warning(f"No risk text to process for {ticker} ({filing_year})")
                return None
                
            # Clean text
            text = self.clean_risk_factor_text(risk_text)
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # Process paragraphs into a DataFrame with metadata
            rows = []
            para_idx = 0
            
            for paragraph in paragraphs:
                # Skip very short paragraphs (likely headers)
                if len(paragraph) < 50:
                    continue
                
                # Split into sentences (improved approach)
                sentences = []
                for s in re.split(r'(?<=[.!?])\s+', paragraph):
                    s = s.strip()
                    if len(s) > 10:  # Skip very short fragments
                        if not s.endswith(('.', '!', '?')):
                            s += '.'
                        sentences.append(s)
                
                for sent_idx, sentence in enumerate(sentences):
                    # Extract sentiment and risk keywords
                    sentiment_score, _ = self.vectorizer.vectorize_sentence(sentence)
                    risk_keywords = self.vectorizer.extract_risk_keywords(sentence)
                    
                    # Create risk factor metadata
                    rows.append({
                        "ticker": ticker,
                        "year": filing_year,
                        "cik": cik,
                        "paragraph_idx": para_idx,
                        "sentence_idx": sent_idx,
                        "sentence": sentence,
                        "word_count": len(sentence.split()),
                        "contains_risk": "risk" in sentence.lower(),
                        "contains_uncertainty": any(term in sentence.lower() for term in [
                            "could", "may", "might", "possible", "potentially", "uncertain", 
                            "risk", "exposure", "adverse", "material", "significant"
                        ]),
                        "sentiment_score": sentiment_score,
                        "risk_keywords": risk_keywords
                    })
                
                para_idx += 1
            
            # Create DataFrame
            df = pd.DataFrame(rows)
            
            # Schema validation
            try:
                # Before validation, ensure np.nan is replaced with None for nullable fields
                df = df.replace({np.nan: None})
                
                # Apply schema validation
                df = risk_factor_schema.validate(df)
            except Exception as e:
                logger.warning(f"Schema validation failed for {ticker} ({filing_year}): {e}")
                # Continue with unvalidated data rather than failing
            
            logger.info(f"Processed {ticker} ({filing_year}): {len(df)} sentences in {para_idx} paragraphs")
            return df
            
        except Exception as e:
            logger.error(f"Error processing text for {ticker} ({filing_year}): {e}")
            return None
    
    def save_dataframe_as_parquet(self, df, ticker, year):
        """Save DataFrame as compressed Parquet with schema enforcement"""
        try:
            if df is None or len(df) == 0:
                logger.warning(f"No data to save for {ticker} ({year})")
                return False
                
            # Create directory
            ticker_dir = PROCESSED_DATA_DIR / ticker
            ticker_dir.mkdir(exist_ok=True)
            
            # Define file path
            parquet_path = ticker_dir / f"{ticker}_{year}_risk.parquet"
            
            # Replace NaN with None for PyArrow compatibility
            df = df.replace({np.nan: None})
            
            # Convert to Arrow Table with schema
            try:
                arrow_table = Table.from_pandas(df, schema=arrow_schema)
            except pa.ArrowInvalid:
                # If schema doesn't match, fall back to inferring schema
                logger.warning(f"Schema mismatch for {ticker} ({year}), using inferred schema")
                arrow_table = pa.Table.from_pandas(df)
            
            # Write with compression
            pq.write_table(
                arrow_table,
                parquet_path,
                compression='ZSTD',
                coerce_timestamps='ms'
            )
            
            logger.info(f"Saved Parquet data for {ticker} ({year})")
            return True
            
        except Exception as e:
            logger.error(f"Error saving Parquet for {ticker} ({year}): {e}")
            # Fall back to CSV if Parquet fails
            try:
                csv_path = ticker_dir / f"{ticker}_{year}_risk.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved CSV backup for {ticker} ({year})")
                return True
            except Exception as csv_e:
                logger.error(f"Error saving CSV backup for {ticker} ({year}): {csv_e}")
                return False
    
    async def process_company_years_async(self, ticker, years):
        """
        Process multiple years for a company concurrently
        
        Args:
            ticker: Company ticker symbol
            years: List of years to process
            
        Returns:
            Dictionary with processing results
        """
        ticker = ticker.upper()
        logger.info(f"Async processing {ticker} for years {years}")
        
        tasks = []
        for year in years:
            tasks.append(self.process_company_filing_async(ticker, year))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = [r for r in results if isinstance(r, dict) and r.get('status') == 'success']
        logger.info(f"Completed async processing for {ticker}: {len(successful)}/{len(years)} years successful")
        
        return {
            "ticker": ticker,
            "years_processed": len(years),
            "years_successful": len(successful),
            "results": results
        }
    
    async def process_company_filing_async(self, ticker, year):
        """
        Process a single company filing asynchronously
        
        Args:
            ticker: Company ticker symbol
            year: Year to process
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Step 1: Try to get filing via API if available
            risk_factors = None
            filing_info = {}
            successful = False
            cik = None
            
            # Use API first if requested
            if self.use_api_first and self.api_calls_remaining > 0:
                logger.info(f"Attempting async API extraction for {ticker} ({year})")
                
                # Get filing URLs via API
                api_filings = self.get_filing_urls_via_api(ticker, year=year)
                
                if api_filings and len(api_filings) > 0:
                    filing_info = api_filings[0]  # Use the first (most recent) filing
                    accession_no = filing_info.get('accession_no')
                    filing_year = filing_info.get('filing_year')
                    cik = filing_info.get('cik')
                    
                    # Try to extract risk factors via API
                    risk_section = await self.async_processor.fetch_api_filing(accession_no)
                    
                    if risk_section and risk_section.get('content'):
                        content = risk_section.get('content')
                        
                        # Check for XBRL
                        xbrl_compliant = bool(re.search(r'<ix:header>', content))
                        if xbrl_compliant:
                            content = self._clean_xbrl(content)
                        
                        risk_factors = content
                        logger.info(f"Successfully extracted via API for {ticker} ({filing_year})")
                        successful = True
            
            # Fall back to scraping if API failed or not used
            if not successful:
                logger.info(f"Falling back to scraping for {ticker} ({year})")
                
                # Get filing URLs via scraping
                scraped_filings = self.get_filing_urls_via_scraping(ticker, year=year)
                
                if scraped_filings and len(scraped_filings) > 0:
                    filing_info = scraped_filings[0]  # Use the first (most recent) filing
                    filing_detail_url = filing_info.get('filing_detail_url')
                    filing_year = filing_info.get('filing_year')
                    cik = filing_info.get('cik')
                    
                    # Get document URL
                    response = await self.async_processor.fetch_filing(filing_detail_url, self.headers)
                    
                    if response:
                        soup = BeautifulSoup(response, 'html.parser')
                        table = soup.find('table', {'class': 'tableFile'})
                        
                        if table:
                            document_url = None
                            for row in table.find_all('tr'):
                                cells = row.find_all('td')
                                if len(cells) >= 3:
                                    doc_link = cells[2].a
                                    if doc_link and (doc_link.text.endswith('.htm') or doc_link.text.endswith('.html')):
                                        if '10-K' in doc_link.text or 'primary_doc' in doc_link.text.lower():
                                            document_url = f"{SEC_BASE_URL}{doc_link['href']}"
                                            break
                            
                            if document_url:
                                # Extract risk factors via scraping
                                doc_content = await self.async_processor.fetch_filing(document_url, self.headers)
                                
                                if doc_content:
                                    # Process with regular extraction methods
                                    result = self.extract_risk_factors_via_scraping(
                                        document_url, ticker, filing_year, filing_info
                                    )
                                    
                                    if result and result.get('content'):
                                        risk_factors = result.get('content')
                                        cik = result.get('cik') or cik
                                        logger.info(f"Successfully extracted via scraping for {ticker} ({filing_year})")
                                        successful = True
            
            # Process the extracted risk factors
            if successful and risk_factors:
                # Save raw content
                company_dir = RAW_DATA_DIR / ticker
                company_dir.mkdir(exist_ok=True)
                
                raw_path = company_dir / f"{ticker}_{filing_year}_10K_risk_factors_raw.txt"
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(risk_factors)
                
                # Process into DataFrame
                df = self.process_filing_to_dataframe(risk_factors, ticker, filing_year, cik)
                
                if df is not None:
                    # Save processed DataFrame as Parquet
                    self.save_dataframe_as_parquet(df, ticker, filing_year)
                    
                    return {
                        "ticker": ticker,
                        "year": filing_year,
                        "cik": cik,
                        "status": "success",
                        "method": "api" if self.use_api_first and self.api_calls_remaining > 0 else "scraping",
                        "sentences": len(df),
                        "xbrl_compliant": bool(re.search(r'<ix:header>', risk_factors)),
                        "dataframe": df
                    }
            
            logger.warning(f"Failed to extract risk factors for {ticker} ({year})")
            return {
                "ticker": ticker,
                "year": year,
                "cik": cik,
                "status": "failed",
                "method": None,
                "sentences": 0,
                "dataframe": None
            }
            
        except Exception as e:
            logger.error(f"Error processing {ticker} ({year}): {e}")
            return {
                "ticker": ticker,
                "year": year,
                "cik": cik if 'cik' in locals() else None,
                "status": "error",
                "error": str(e),
                "dataframe": None
            }
    
    def process_company(self, ticker, start_year=2018, end_year=None):
        """
        Process all 10-K filings for a company across years
        
        Args:
            ticker: Company ticker symbol
            start_year: Earliest year to process
            end_year: Latest year to process
            
        Returns:
            Dictionary with processing results
        """
        if end_year is None:
            end_year = datetime.now().year
            
        ticker = ticker.upper()
        logger.info(f"Processing {ticker} from {start_year} to {end_year}")
        
        # Process each year (either sequentially or async)
        years = list(range(start_year, end_year + 1))
        
        try:
            # Run async processing
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        results = loop.run_until_complete(self.process_company_years_async(ticker, years))
        
        # Filter successful results
        successful_results = [r for r in results.get('results', []) 
                             if isinstance(r, dict) and r.get('status') == 'success']
        
        # Get DataFrames from successful results
        dataframes = [r.get('dataframe') for r in successful_results if r.get('dataframe') is not None]
        
        # Analyze results across years if we have data
        if dataframes:
            self.analyze_risk_factors(ticker, dataframes)
        
        # Create summary
        summary = {
            "ticker": ticker,
            "years_processed": end_year - start_year + 1,
            "years_successful": len(successful_results),
            "years_covered": sorted([r.get('year') for r in successful_results]),
            "total_sentences": sum(r.get('sentences', 0) for r in successful_results),
            "api_calls_used": MAX_API_CALLS - self.api_calls_remaining if self.use_api_first else 0,
            "status": "success" if successful_results else "failed",
            "xbrl_compliance": {r.get('year'): r.get('xbrl_compliant', False) for r in successful_results},
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary
        summary_path = PROCESSED_DATA_DIR / f"{ticker}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        logger.info(f"Completed processing for {ticker}: {summary['years_successful']}/{summary['years_processed']} years successful")
        return summary
    
    def analyze_risk_factors(self, ticker, dataframes):
        """
        Analyze risk factors across years for a company
        
        Args:
            ticker: Company ticker symbol
            dataframes: List of DataFrames with risk factor data
            
        Returns:
            DataFrame with analysis results
        """
        try:
            # Combine all dataframes
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            # Get unique years
            years = sorted(combined_df['year'].unique())
            
            # Calculate statistics for each year
            stats = []
            for year in years:
                year_df = combined_df[combined_df['year'] == year]
                
                # Get CIK (should be the same for all rows)
                cik = year_df['cik'].iloc[0] if not year_df['cik'].isna().all() else None
                
                # Risk keyword analysis
                all_keywords = []
                for keywords in year_df['risk_keywords']:
                    if keywords:
                        all_keywords.extend(keywords)
                
                keyword_counts = pd.Series(all_keywords).value_counts().to_dict()
                top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                
                year_stats = {
                    'ticker': ticker,
                    'year': year,
                    'cik': cik,
                    'total_sentences': len(year_df),
                    'total_paragraphs': year_df['paragraph_idx'].nunique(),
                    'avg_sentence_length': year_df['word_count'].mean(),
                    'risk_sentence_count': year_df['contains_risk'].sum(),
                    'risk_sentence_pct': (year_df['contains_risk'].sum() / len(year_df)) * 100 if len(year_df) > 0 else 0,
                    'uncertainty_sentence_count': year_df['contains_uncertainty'].sum(),
                    'uncertainty_sentence_pct': (year_df['contains_uncertainty'].sum() / len(year_df)) * 100 if len(year_df) > 0 else 0,
                    'avg_sentiment_score': year_df['sentiment_score'].mean(),
                    'top_risk_keywords': dict(top_keywords)
                }
                
                stats.append(year_stats)
            
            # Create DataFrame from statistics
            stats_df = pd.DataFrame(stats)
            
            # Calculate year-over-year changes
            if len(years) > 1:
                # Sort by year
                stats_df = stats_df.sort_values('year')
                
                # Calculate deltas
                stats_df['delta_sentences'] = stats_df['total_sentences'].diff()
                stats_df['delta_paragraphs'] = stats_df['total_paragraphs'].diff()
                stats_df['delta_avg_length'] = stats_df['avg_sentence_length'].diff()
                stats_df['delta_risk_pct'] = stats_df['risk_sentence_pct'].diff()
                stats_df['delta_uncertainty_pct'] = stats_df['uncertainty_sentence_pct'].diff()
                stats_df['delta_sentiment'] = stats_df['avg_sentiment_score'].diff()
                
                # Calculate percent changes
                stats_df['pct_change_sentences'] = stats_df['total_sentences'].pct_change() * 100
                stats_df['pct_change_risk'] = stats_df['risk_sentence_count'].pct_change() * 100
                stats_df['pct_change_uncertainty'] = stats_df['uncertainty_sentence_count'].pct_change() * 100
            
            # Save analysis results
            analysis_dir = ANALYSIS_DIR / ticker
            analysis_dir.mkdir(exist_ok=True)
            
            # Save as CSV and Parquet
            analysis_csv_path = analysis_dir / f"{ticker}_risk_analysis.csv"
            stats_df.to_csv(analysis_csv_path, index=False)
            
            # Save as Parquet (more efficient for larger datasets)
            analysis_parquet_path = analysis_dir / f"{ticker}_risk_analysis.parquet"
            pq.write_table(
                pa.Table.from_pandas(stats_df.drop(columns=['top_risk_keywords'])),  # Complex columns need special handling
                analysis_parquet_path,
                compression='ZSTD'
            )
            
            # Save top keywords separately as JSON
            keywords_path = analysis_dir / f"{ticker}_top_keywords.json"
            with open(keywords_path, 'w') as f:
                json.dump({
                    str(year): stats_df[stats_df['year'] == year]['top_risk_keywords'].iloc[0]
                    for year in years
                }, f, indent=4)
            
            # Create visualization (optional)
            try:
                import matplotlib.pyplot as plt
                
                # Set style
                plt.style.use('ggplot')
                
                # Yearly trends plot
                fig = plt.figure(figsize=(12, 10))
                
                # Plot 1: Risk and Uncertainty Percentage
                ax1 = fig.add_subplot(2, 2, 1)
                ax1.plot(stats_df['year'], stats_df['risk_sentence_pct'], marker='o', label='Risk', linewidth=2)
                ax1.plot(stats_df['year'], stats_df['uncertainty_sentence_pct'], marker='s', label='Uncertainty', linewidth=2)
                ax1.set_title(f'{ticker} Risk Factor Analysis', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Year')
                ax1.set_ylabel('Percentage of Sentences')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Total Sentences and Paragraphs
                ax2 = fig.add_subplot(2, 2, 2)
                ax2.bar(stats_df['year'].astype(str), stats_df['total_sentences'], alpha=0.7, label='Sentences')
                ax2.bar(stats_df['year'].astype(str), stats_df['total_paragraphs'], alpha=0.5, label='Paragraphs')
                ax2.set_title('Document Size', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Year')
                ax2.set_ylabel('Count')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Average Sentiment Score
                ax3 = fig.add_subplot(2, 2, 3)
                ax3.plot(stats_df['year'], stats_df['avg_sentiment_score'], marker='d', color='green', linewidth=2)
                ax3.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # Zero line
                ax3.set_title('Average Sentiment (FinBERT)', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Year')
                ax3.set_ylabel('Sentiment Score (-1 to 1)')
                ax3.grid(True, alpha=0.3)
                
                # Plot 4: Year-over-Year Changes (if applicable)
                if len(years) > 1:
                    ax4 = fig.add_subplot(2, 2, 4)
                    year_labels = stats_df['year'][1:].astype(str)
                    ax4.bar(year_labels, stats_df['pct_change_risk'][1:], alpha=0.7, label='Risk')
                    ax4.bar(year_labels, stats_df['pct_change_uncertainty'][1:], alpha=0.5, label='Uncertainty')
                    ax4.set_title('YoY % Change in Risk Factors', fontsize=12, fontweight='bold')
                    ax4.set_xlabel('Year')
                    ax4.set_ylabel('Percent Change')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                plt_path = analysis_dir / f"{ticker}_risk_analysis_plot.png"
                plt.savefig(plt_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Created visualization for {ticker}")
                
            except Exception as e:
                logger.warning(f"Could not create visualization for {ticker}: {e}")
            
            logger.info(f"Completed risk factor analysis for {ticker}")
            return stats_df
            
        except Exception as e:
            logger.error(f"Error analyzing risk factors for {ticker}: {e}")
            return None


async def process_multiple_companies(tickers, start_year=2018, end_year=None, max_concurrent=5):
    """
    Process multiple companies concurrently with a limit on concurrency
    
    Args:
        tickers: List of ticker symbols
        start_year: Earliest year to retrieve
        end_year: Latest year to retrieve
        max_concurrent: Maximum number of concurrent processes
        
    Returns:
        Dictionary with processing results
    """
    if end_year is None:
        end_year = datetime.now().year
    
    scraper = SECRiskFactorScraper(api_key=SEC_API_KEY, use_api_first=True if SEC_API_KEY else False)
    results = {}
    
    # Process in batches to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(ticker):
        async with semaphore:
            try:
                years = list(range(start_year, end_year + 1))
                result = await scraper.process_company_years_async(ticker, years)
                return ticker, result
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                return ticker, {"ticker": ticker, "status": "error", "error": str(e)}
    
    # Create tasks
    tasks = [process_with_semaphore(ticker.upper()) for ticker in tickers]
    
    # Run and collect results
    for completed_task in asyncio.as_completed(tasks):
        ticker, result = await completed_task
        results[ticker] = result
        logger.info(f"Completed {ticker}: {result.get('years_successful', 0)}/{result.get('years_processed', 0)} years successful")
    
    # Log API usage
    if SEC_API_KEY:
        logger.info(f"API calls remaining: {scraper.api_calls_remaining} (out of {MAX_API_CALLS})")
    
    return results


def main():
    """Main execution function"""
    # List of companies to process
    companies = [
        {"ticker": "AAPL", "name": "Apple Inc."},
        # {"ticker": "MSFT", "name": "Microsoft Corporation"},
        # {"ticker": "GOOGL", "name": "Alphabet Inc."},
        # {"ticker": "AMZN", "name": "Amazon.com, Inc."},
        # {"ticker": "META", "name": "Meta Platforms, Inc."}
    ]
    
    # Extract just the tickers
    tickers = [company["ticker"] for company in companies]
    
    test_start_year = 2022 # Adjust this to limit API calls
    test_end_year = 2023   # Adjust this to limit API calls
    
    # Run async processing with event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    results = loop.run_until_complete(
        process_multiple_companies(
            tickers,
            start_year=test_start_year,
            end_year=test_end_year,
            max_concurrent=2 # Lower concurrency for testing if needed
        )
    )
    
    # Save overall results
    results_path = BASE_DIR / f"sec_risk_factors_results_{datetime.now().strftime('%Y%m%d')}.json"
    
    # Clean results for JSON serialization (remove DataFrames)
    clean_results = {}
    for ticker, result in results.items():
        if 'results' in result:
            clean_result = result.copy()
            clean_result['results'] = [{k: v for k, v in r.items() if k != 'dataframe'} 
                                      for r in result['results'] if isinstance(r, dict)]
            clean_results[ticker] = clean_result
        else:
            clean_results[ticker] = result
    
    with open(results_path, 'w') as f:
        json.dump(clean_results, f, indent=4)
    
    logger.info(f"SEC Risk Factor extraction complete. Processed {len(companies)} companies.")
    
    # Print summary statistics
    successful_companies = sum(1 for r in results.values() 
                              if isinstance(r, dict) and r.get('years_successful', 0) > 0)
    
    total_years_processed = sum(r.get('years_processed', 0) for r in results.values() if isinstance(r, dict))
    total_years_successful = sum(r.get('years_successful', 0) for r in results.values() if isinstance(r, dict))
    
    logger.info(f"Success rate: {successful_companies}/{len(companies)} companies ({successful_companies/len(companies)*100:.1f}%)")
    logger.info(f"Success rate: {total_years_successful}/{total_years_processed} years ({total_years_successful/total_years_processed*100:.1f}%)")


if __name__ == "__main__":
    main()