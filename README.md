# NSE Bhavcopy Data Downloader

Automated script to download full bhavcopy and security deliverable data from NSE India website.

## Features

- ✅ Automated navigation through NSE website with calendar-based date selection
- ✅ Configurable date range via command-line arguments
- ✅ Weekly browser session batching (Monday-Friday) for improved performance
- ✅ Intelligent operation skipping - full navigation only on first day of week
- ✅ Automatic weekend skipping with logged entries
- ✅ Rotating browser user agents for each download
- ✅ Optimized wait times (5 seconds per selector)
- ✅ 60-second download wait with retry mechanism
- ✅ Random sleep intervals between downloads (3-7 seconds)
- ✅ Comprehensive logging to file and console (stored in `logs/` folder)
- ✅ CSV summary table with download status, file size, and pandas shape analysis
- ✅ Tracks failed downloads with weekday information

## Prerequisites

- Python 3.8 or higher
- Google Chrome browser installed
- ChromeDriver (will be managed automatically by selenium)

## Installation

1. Install required packages:

```powershell
pip install -r requirements.txt
```

Or with uv:

```powershell
uv sync
```

## Usage

### Basic Usage (with defaults)

```powershell
python download_nse_bhavcopy.py
```

This will download data from July 1-10, 2025 to `data/202507/`

### Custom Date Range

```powershell
python download_nse_bhavcopy.py --start-date 2025-06-01 --end-date 2025-06-15
```

### Custom Output Directory

```powershell
python download_nse_bhavcopy.py --start-date 2025-07-01 --end-date 2025-07-10 --output-dir data/custom_folder
```

### Using uv

```powershell
uv run python download_nse_bhavcopy.py --start-date 2025-07-01 --end-date 2025-07-10
```

### Command-line Arguments

- `--start-date`: Start date in YYYY-MM-DD format (default: 2025-07-01)
- `--end-date`: End date in YYYY-MM-DD format (default: 2025-07-10)
- `--output-dir`: Output directory (default: data/YYYYMM based on start date)

### Examples

Download data for entire June 2025:

```powershell
python download_nse_bhavcopy.py --start-date 2025-06-01 --end-date 2025-06-30
```

Download single day:

```powershell
python download_nse_bhavcopy.py --start-date 2025-07-15 --end-date 2025-07-15
```

## Output

The script creates the following:

### In the output directory (e.g., `data/202507/`)

1. **Downloaded CSV files**: `sec_bhavdata_full_DDMMYYYY.csv`

### In the `logs/` folder

1. **download_log_YYYYMMDD_YYYYMMDD.txt**: Detailed log of all operations
2. **download_summary_YYYYMMDD_YYYYMMDD.csv**: Summary table with columns:
   - Date
   - Weekday
   - Status (Success/Failed/Skipped)
   - Filename
   - Error (if any)
   - File_Size_KB
   - Rows (from pandas shape analysis)
   - Columns (from pandas shape analysis)

## Configuration

You can modify the following variables in the script:

- `SLEEP_MIN` and `SLEEP_MAX`: Sleep interval range between downloads (default: 3-7 seconds)
- `USER_AGENTS`: List of browser user agents to rotate

## Notes

- The script uses Selenium WebDriver with Chrome to automate browser interactions
- Direct URL navigation to archives page: `https://www.nseindia.com/all-reports#cr_equity_archives`
- Calendar interaction using `gj-picker` class for date selection
- Weekly browser session batching: new session starts Monday, reused through Friday
- First day of week performs full navigation, search, and checkbox selection
- Subsequent days in the same week skip directly to date selection for speed
- Weekends (Saturday/Sunday) are automatically skipped with logged entries
- User agents are rotated for each weekly session
- Sleep intervals are randomized to avoid triggering rate limits
- Failed downloads are logged with their weekday for reference
- Download waits up to 60 seconds with 2-second retry interval
- File analysis performed after all downloads complete using pandas
