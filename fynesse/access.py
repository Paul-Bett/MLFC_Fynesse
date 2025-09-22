"""
Access module for the fynesse framework.

This module handles data access functionality including:
- Data loading from various sources (web, local files, databases)
- Legal compliance (intellectual property, privacy rights)
- Ethical considerations for data usage
- Error handling for access issues

Legal and ethical considerations are paramount in data access.
Ensure compliance with e.g. .GDPR, intellectual property laws, and ethical guidelines.

Best Practice on Implementation
===============================

1. BASIC ERROR HANDLING:
   - Use try/except blocks to catch common errors
   - Provide helpful error messages for debugging
   - Log important events for troubleshooting

2. WHERE TO ADD ERROR HANDLING:
   - File not found errors when loading data
   - Network errors when downloading from web
   - Permission errors when accessing files
   - Data format errors when parsing files

3. SIMPLE LOGGING:
   - Use print() statements for basic logging
   - Log when operations start and complete
   - Log errors with context information
   - Log data summary information

4. EXAMPLE PATTERNS:
   
   Basic error handling:
   try:
       df = pd.read_csv('data.csv')
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
   
   With logging:
   print("Loading data from data.csv...")
   try:
       df = pd.read_csv('data.csv')
       print(f"Successfully loaded {len(df)} rows of data")
       return df
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
"""

from typing import Any, Union
import pandas as pd
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def data() -> Union[pd.DataFrame, None]:
    """
    Read the data from the web or local file, returning structured format such as a data frame.

    IMPLEMENTATION GUIDE
    ====================

    1. REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING CODE:
       - Load data from your specific sources
       - Handle common errors (file not found, network issues)
       - Validate that data loaded correctly
       - Return the data in a useful format

    2. ADD ERROR HANDLING:
       - Use try/except blocks for file operations
       - Check if data is empty or corrupted
       - Provide helpful error messages

    3. ADD BASIC LOGGING:
       - Log when you start loading data
       - Log success with data summary
       - Log errors with context

    4. EXAMPLE IMPLEMENTATION:
       try:
           print("Loading data from data.csv...")
           df = pd.read_csv('data.csv')
           print(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
           return df
       except FileNotFoundError:
           print("Error: data.csv file not found")
           return None
       except Exception as e:
           print(f"Error loading data: {e}")
           return None

    Returns:
        DataFrame or other structured data format
    """
    logger.info("Starting data access operation")

    try:
        # IMPLEMENTATION: Replace this with your actual data loading code
        # Example: Load data from a CSV file
        logger.info("Loading data from data.csv")
        df = pd.read_csv("data.csv")

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(
            f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except FileNotFoundError:
        logger.error("Data file not found: data.csv")
        print("Error: Could not find data.csv file. Please check the file path.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        print(f"Error loading data: {e}")
        return None


def load_text_log(path):
    """Load raw text logfile; return list of lines."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines

def parse_counts_from_loglines(lines,
                              timestamp_regex=r'Processed (\d{14})-snapshot\.jpg', # Updated regex for YYYYMMDDhhmmss
                              count_regex=r'Insect count: (\d+)', # Updated regex for "Insect count: "
                              timestamp_fmt='%Y%m%d%H%M%S'): # Updated format string
    """
    Parse timestamp and insect count from each log line.
    Returns a DataFrame with columns ['timestamp','count','raw_line'].
    Regexes can be tuned to match your logfile format.
    """
    records = []
    for ln in lines:
        # timestamp
        ts_match = re.search(timestamp_regex, ln)
        if not ts_match:
            # try ISO-like or epoch parse fallback
            # skip lines without timestamp
            continue
        ts_str = ts_match.group(1)
        try:
            ts = datetime.strptime(ts_str, timestamp_fmt)
        except Exception:
            # try parsing with pandas (more tolerant)
            ts = pd.to_datetime(ts_str, errors='coerce', format=timestamp_fmt) # Added format to pandas parse
            if pd.isna(ts):
                continue
            ts = ts.to_pydatetime()
        # count
        c = None
        m = re.search(count_regex, ln, flags=re.I) # Use the updated count regex
        if m:
            c = int(m.group(1))
        else:
            # Fallback to broader count regex if specific one fails
            m2 = re.search(r'(\bcount[:= ]*\s*(\d+)\b|\b(\d+)\s*(?:insects|moths|bugs)?\b)', ln, flags=re.I)
            if m2:
                c = int(m2.group(2) or m2.group(3)) # Capture group 2 or 3 from the broader regex
        if c is None:
            # if no count found, consider NaN (we'll drop or impute later)
            c = np.nan
        records.append({'timestamp': ts, 'count': c, 'raw_line': ln})
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def load_weather_csv(path, time_col='timestamp', time_fmt=None):
    """
    Load weather CSV. Attempts to parse the time column to datetime.
    Returns DataFrame with datetime index.
    """
    df = pd.read_csv(path)
    if time_col not in df.columns:
        # try common names
        for c in df.columns:
            if 'time' in c.lower() or 'date' in c.lower():
                time_col = c
                break
    if time_fmt:
        df[time_col] = pd.to_datetime(df[time_col], format=time_fmt, errors='coerce')
    else:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])
    df = df.set_index(time_col).sort_index()
    return df
