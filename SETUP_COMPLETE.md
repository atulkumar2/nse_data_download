# NSE Bhavcopy Data Downloader - Environment Setup Complete

## Environment Created with UV

Your Python environment has been successfully set up using `uv`.

### Virtual Environment Location

`.venv/` (in the project root)

### Installed Packages

- selenium 4.38.0
- pandas 2.3.3
- webdriver-manager 4.0.2
- All dependencies

---

## Quick Start

### 1. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Run the download script

```powershell
python download_nse_bhavcopy.py
```

### 3. Or use uv to run directly (without activation)

```powershell
uv run python download_nse_bhavcopy.py
```

---

## Project Structure

```text
nse_data_download/
├── .venv/                          # Virtual environment (created by uv)
├── data/                           # Data directory
│   └── 202507/                     # July 2025 downloads
│       ├── sec_bhavdata_full_*.csv # Downloaded files
│       ├── download_log.txt        # Detailed logs
│       └── download_summary.csv    # Summary table
├── download_nse_bhavcopy.py        # Main script
├── pyproject.toml                  # Project configuration
├── requirements.txt                # Package list
└── README.md                       # Documentation
```

---

## Managing the Environment

### Install new packages

```powershell
uv pip install package-name
```

### Update all packages

```powershell
uv sync
```

### Add a package to pyproject.toml

Edit the `dependencies` list in `pyproject.toml`, then run:

```powershell
uv sync
```

---

## Running the Script

The script will:

- Download NSE bhavcopy data for July 1-10, 2025
- Save files to `data/202507/`
- Rotate through 7 different browser user agents
- Add 3-7 second random delays between downloads
- Create detailed logs and summary table
- Track failed downloads with weekday information

**Prerequisites**: Make sure Google Chrome browser is installed on your system.

---

Enjoy your automated NSE data downloads! 🚀
