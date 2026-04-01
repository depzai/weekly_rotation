"""
Portfolio Rotation Strategy
============================
Runs weekly (Friday after close) and produces THREE ranked stock lists:

  1. MOMENTUM       -- top stocks by 20-day price return
  2. MEAN REVERSION -- most oversold stocks (low RSI) in strong sectors
  3. RELATIVE STRENGTH -- stocks outperforming SPY over 20 days

Each list is filtered by sector strength first:
  - Ranks all 11 sector ETFs by 20-day return
  - Only considers stocks in the TOP 5 sectors

Output: three tabs in Google Sheets ("ranging"):
  - rotation_momentum
  - rotation_mean_reversion
  - rotation_relative_strength

GitHub Actions secrets required:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
    GSPREAD_SA_KEY_JSON

Setup:
    pip install alpaca-trade-api yfinance pandas numpy requests gspread google-auth lxml html5lib
"""

import os
import time
import tempfile
import logging
import requests
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    raise ImportError("pip install gspread google-auth")


# ===========================================================================
# LOGGING
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# CONFIG
# ===========================================================================

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY",    "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
GSPREAD_SA        = os.getenv("GSPREAD_SA_KEY_JSON", "")

# Google Sheets
GSHEET_NAME = "ranging"
TAB_MOMENTUM   = "rotation_momentum"
TAB_REVERSION  = "rotation_mean_reversion"
TAB_RELSTRENGTH= "rotation_relative_strength"
TAB_SECTORS    = "rotation_sectors"

GSCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Strategy params
LOOKBACK_DAYS    = 60     # fetch 60 days of data for calculations
MOMENTUM_PERIOD  = 20     # 20-day return for momentum signal
RS_PERIOD        = 20     # 20-day return vs SPY for relative strength
RSI_PERIOD       = 14     # RSI period for mean reversion
RSI_THRESHOLD    = 45     # RSI below this = oversold candidate
TOP_SECTORS      = 5      # only consider stocks in top N sectors
TOP_N_STOCKS     = 20     # stocks per list
MIN_PRICE        = 10.0   # skip penny stocks
MIN_AVG_VOLUME   = 500_000

# Sector ETFs
SECTOR_ETFS = {
    "XLK" : "Technology",
    "XLV" : "Healthcare",
    "XLE" : "Energy",
    "XLF" : "Financials",
    "XLI" : "Industrials",
    "XLP" : "Consumer Staples",
    "XLY" : "Consumer Discretionary",
    "XLB" : "Materials",
    "XLU" : "Utilities",
    "XLRE": "Real Estate",
    "XLC" : "Communication",
}

# Map tickers to sectors (used to filter stocks by top sectors)
# yfinance provides this via ticker.info but it's slow for 500 stocks
# We use a hardcoded map for speed -- covers most S&P 500 stocks
SECTOR_MAP = {
    # Technology
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","AVGO":"XLK","AMD":"XLK","INTC":"XLK",
    "QCOM":"XLK","TXN":"XLK","AMAT":"XLK","LRCX":"XLK","KLAC":"XLK","ADI":"XLK",
    "MRVL":"XLK","MU":"XLK","NXPI":"XLK","ON":"XLK","MPWR":"XLK","SWKS":"XLK",
    "ADBE":"XLK","CRM":"XLK","NOW":"XLK","INTU":"XLK","PANW":"XLK","CRWD":"XLK",
    "FTNT":"XLK","ZS":"XLK","DDOG":"XLK","SNOW":"XLK","PLTR":"XLK","DELL":"XLK",
    "HPE":"XLK","NTAP":"XLK","STX":"XLK","WDC":"XLK","GDDY":"XLK","CDW":"XLK",
    "CDNS":"XLK","SNPS":"XLK","ANSS":"XLK","PTC":"XLK","TDY":"XLK","TER":"XLK",
    # Communication
    "GOOGL":"XLC","GOOG":"XLC","META":"XLC","NFLX":"XLC","DIS":"XLC","CMCSA":"XLC",
    "T":"XLC","VZ":"XLC","TMUS":"XLC","CHTR":"XLC","WBD":"XLC","FOXA":"XLC","FOX":"XLC",
    "MTCH":"XLC","EA":"XLC","TTWO":"XLC","AKAM":"XLC","IPG":"XLC","OMC":"XLC",
    # Consumer Discretionary
    "AMZN":"XLY","TSLA":"XLY","HD":"XLY","MCD":"XLY","NKE":"XLY","LOW":"XLY",
    "SBUX":"XLY","TJX":"XLY","BKNG":"XLY","MAR":"XLY","HLT":"XLY","GM":"XLY",
    "F":"XLY","ORLY":"XLY","AZO":"XLY","ROST":"XLY","EBAY":"XLY","EXPE":"XLY",
    "ABNB":"XLY","LVS":"XLY","MGM":"XLY","WYNN":"XLY","YUM":"XLY","DPZ":"XLY",
    # Consumer Staples
    "WMT":"XLP","COST":"XLP","PG":"XLP","KO":"XLP","PEP":"XLP","PM":"XLP",
    "MO":"XLP","MDLZ":"XLP","CL":"XLP","KMB":"XLP","GIS":"XLP","K":"XLP",
    "CAG":"XLP","SJM":"XLP","HSY":"XLP","MKC":"XLP","CPB":"XLP","TAP":"XLP",
    # Energy
    "XOM":"XLE","CVX":"XLE","COP":"XLE","EOG":"XLE","SLB":"XLE","MPC":"XLE",
    "PSX":"XLE","VLO":"XLE","OXY":"XLE","DVN":"XLE","HAL":"XLE","APA":"XLE",
    "FANG":"XLE","HES":"XLE","BKR":"XLE","CTRA":"XLE","MRO":"XLE","EQT":"XLE",
    # Financials
    "JPM":"XLF","BAC":"XLF","WFC":"XLF","GS":"XLF","MS":"XLF","BLK":"XLF",
    "SCHW":"XLF","AXP":"XLF","COF":"XLF","USB":"XLF","TFC":"XLF","PNC":"XLF",
    "C":"XLF","BK":"XLF","STT":"XLF","MTB":"XLF","FITB":"XLF","HBAN":"XLF",
    "RF":"XLF","CFG":"XLF","SPGI":"XLF","MCO":"XLF","ICE":"XLF","CME":"XLF",
    "CB":"XLF","AON":"XLF","MMC":"XLF","AIG":"XLF","ALL":"XLF","PGR":"XLF",
    # Healthcare
    "UNH":"XLV","LLY":"XLV","JNJ":"XLV","ABT":"XLV","MRK":"XLV","TMO":"XLV",
    "DHR":"XLV","ABBV":"XLV","BMY":"XLV","PFE":"XLV","AMGN":"XLV","GILD":"XLV",
    "ISRG":"XLV","SYK":"XLV","BSX":"XLV","MDT":"XLV","EW":"XLV","BDX":"XLV",
    "IQV":"XLV","VRTX":"XLV","REGN":"XLV","BIIB":"XLV","MRNA":"XLV","IDXX":"XLV",
    "HCA":"XLV","ELV":"XLV","CI":"XLV","HUM":"XLV","MOH":"XLV","CVS":"XLV",
    "MCK":"XLV","COR":"XLV","CAH":"XLV","HSIC":"XLV","HOLX":"XLV","PODD":"XLV",
    # Industrials
    "GE":"XLI","CAT":"XLI","HON":"XLI","RTX":"XLI","LMT":"XLI","NOC":"XLI",
    "GD":"XLI","BA":"XLI","DE":"XLI","ETN":"XLI","EMR":"XLI","ITW":"XLI",
    "PH":"XLI","CMI":"XLI","DOV":"XLI","XYL":"XLI","AME":"XLI","FAST":"XLI",
    "GWW":"XLI","MSI":"XLI","LDOS":"XLI","HII":"XLI","TXT":"XLI","HWM":"XLI",
    # Materials
    "LIN":"XLB","APD":"XLB","SHW":"XLB","FCX":"XLB","NEM":"XLB","NUE":"XLB",
    "STLD":"XLB","CF":"XLB","MOS":"XLB","ALB":"XLB","CE":"XLB","EMN":"XLB",
    "DD":"XLB","DOW":"XLB","LYB":"XLB","PPG":"XLB","ECL":"XLB","IFF":"XLB",
    # Utilities
    "NEE":"XLU","DUK":"XLU","SO":"XLU","D":"XLU","AEP":"XLU","EXC":"XLU",
    "XEL":"XLU","ED":"XLU","ETR":"XLU","EIX":"XLU","WEC":"XLU","ES":"XLU",
    "DTE":"XLU","CMS":"XLU","NI":"XLU","AEE":"XLU","LNT":"XLU","EVRG":"XLU",
    # Real Estate
    "PLD":"XLRE","AMT":"XLRE","CCI":"XLRE","EQIX":"XLRE","PSA":"XLRE","SPG":"XLRE",
    "O":"XLRE","DLR":"XLRE","WELL":"XLRE","EXR":"XLRE","AVB":"XLRE","EQR":"XLRE",
    "VTR":"XLRE","ARE":"XLRE","MAA":"XLRE","UDR":"XLRE","CPT":"XLRE","REG":"XLRE",
}


# ===========================================================================
# GOOGLE SHEETS
# ===========================================================================

def _gs_client():
    if not GSPREAD_SA:
        raise EnvironmentError("GSPREAD_SA_KEY_JSON not set")
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    tf.write(GSPREAD_SA)
    tf.flush()
    creds = Credentials.from_service_account_file(tf.name, scopes=GSCOPES)
    return gspread.authorize(creds)


def _ensure_tab(ss, tab_name, headers):
    try:
        ws = ss.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=tab_name, rows=2000, cols=len(headers))
        log.info(f"  Created tab: {tab_name}")
    existing = ws.row_values(1)
    if existing != headers:
        if existing:
            ws.delete_rows(1)
        ws.insert_row(headers, index=1, value_input_option="RAW")
    return ws


def write_ranked_list(ss, tab_name: str, headers: list, rows: list, run_ts: str):
    """Overwrites the tab with a fresh ranked list each week."""
    ws = _ensure_tab(ss, tab_name, headers)
    # Clear all data rows (keep header)
    all_vals = ws.get_all_values()
    if len(all_vals) > 1:
        ws.delete_rows(2, len(all_vals))
    # Write new rows
    if rows:
        ws.append_rows(rows, value_input_option="RAW")
    log.info(f"  Written {len(rows)} rows to '{tab_name}'")


# ===========================================================================
# UNIVERSE
# ===========================================================================

def fetch_sp500() -> list:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; scanner/1.0)"}
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=15,
        )
        r.raise_for_status()
        tickers = pd.read_html(r.text)[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info(f"  S&P 500: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        log.warning(f"  S&P 500 fetch failed: {e}")
    return []


# ===========================================================================
# DATA
# ===========================================================================

def fetch_prices(tickers: list, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Returns a DataFrame of adjusted closing prices.
    Columns = tickers, index = dates.
    """
    end   = datetime.today()
    start = end - timedelta(days=days)
    log.info(f"  Downloading {len(tickers)} tickers ...")

    chunk_size = 200
    frames = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            raw = yf.download(
                chunk,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"]
            else:
                closes = raw[["Close"]].rename(columns={"Close": chunk[0]})
            frames.append(closes)
        except Exception as e:
            log.warning(f"  Chunk {i//chunk_size+1} failed: {e}")
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices = prices.dropna(how="all")
    log.info(f"  Price data: {prices.shape[0]} days x {prices.shape[1]} tickers")
    return prices


def fetch_volumes(tickers: list, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Returns average daily volume per ticker."""
    end   = datetime.today()
    start = end - timedelta(days=days)
    vols  = {}

    chunk_size = 200
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            raw = yf.download(
                chunk,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                vol_df = raw["Volume"]
            else:
                vol_df = raw[["Volume"]].rename(columns={"Volume": chunk[0]})
            for t in vol_df.columns:
                vols[t] = float(vol_df[t].mean())
        except Exception:
            pass
        time.sleep(0.3)

    return pd.Series(vols)


# ===========================================================================
# SECTOR RANKING
# ===========================================================================

def rank_sectors(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks sector ETFs by 20-day return.
    Returns DataFrame sorted best to worst.
    """
    sector_tickers = list(SECTOR_ETFS.keys())
    available = [t for t in sector_tickers if t in prices.columns]

    rows = []
    for etf in available:
        series = prices[etf].dropna()
        if len(series) < MOMENTUM_PERIOD + 1:
            continue
        ret_20d = (series.iloc[-1] / series.iloc[-MOMENTUM_PERIOD] - 1) * 100
        rows.append({
            "etf"     : etf,
            "sector"  : SECTOR_ETFS[etf],
            "return_20d": round(ret_20d, 2),
        })

    df = pd.DataFrame(rows).sort_values("return_20d", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # rank starts at 1
    return df


# ===========================================================================
# SIGNAL 1: MOMENTUM
# ===========================================================================

def compute_momentum(
    prices    : pd.DataFrame,
    volumes   : pd.Series,
    top_sector_etfs: list,
) -> pd.DataFrame:
    """
    Ranks stocks by 20-day price return.
    Only includes stocks in top sectors.
    """
    rows = []
    for ticker in prices.columns:
        if ticker in SECTOR_ETFS:
            continue  # skip ETFs

        # Sector filter
        sector_etf = SECTOR_MAP.get(ticker)
        if sector_etf and sector_etf not in top_sector_etfs:
            continue

        series = prices[ticker].dropna()
        if len(series) < MOMENTUM_PERIOD + 1:
            continue

        last_price = float(series.iloc[-1])
        if last_price < MIN_PRICE:
            continue

        avg_vol = float(volumes.get(ticker, 0))
        if avg_vol < MIN_AVG_VOLUME:
            continue

        ret_20d  = (series.iloc[-1] / series.iloc[-MOMENTUM_PERIOD] - 1) * 100
        ret_5d   = (series.iloc[-1] / series.iloc[-5] - 1) * 100 if len(series) >= 6 else 0

        rows.append({
            "ticker"    : ticker,
            "sector"    : SECTOR_ETFS.get(sector_etf, "Unknown") if sector_etf else "Unknown",
            "price"     : round(last_price, 2),
            "return_20d": round(float(ret_20d), 2),
            "return_5d" : round(float(ret_5d), 2),
            "avg_volume": int(avg_vol),
        })

    df = pd.DataFrame(rows).sort_values("return_20d", ascending=False).head(TOP_N_STOCKS)
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    return df


# ===========================================================================
# SIGNAL 2: MEAN REVERSION
# ===========================================================================

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    """Compute RSI for the latest bar."""
    delta  = series.diff().dropna()
    gain   = delta.clip(lower=0).rolling(period).mean()
    loss   = (-delta.clip(upper=0)).rolling(period).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi    = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def compute_mean_reversion(
    prices    : pd.DataFrame,
    volumes   : pd.Series,
    top_sector_etfs: list,
) -> pd.DataFrame:
    """
    Finds oversold stocks (RSI < RSI_THRESHOLD) in strong sectors.
    Lower RSI = more oversold = ranked higher.
    Only looks at stocks in top sectors -- avoids catching falling knives
    in weak sectors.
    """
    rows = []
    for ticker in prices.columns:
        if ticker in SECTOR_ETFS:
            continue

        # Sector filter -- mean reversion only in strong sectors
        sector_etf = SECTOR_MAP.get(ticker)
        if sector_etf and sector_etf not in top_sector_etfs:
            continue

        series = prices[ticker].dropna()
        if len(series) < RSI_PERIOD + 5:
            continue

        last_price = float(series.iloc[-1])
        if last_price < MIN_PRICE:
            continue

        avg_vol = float(volumes.get(ticker, 0))
        if avg_vol < MIN_AVG_VOLUME:
            continue

        rsi      = compute_rsi(series)
        ret_20d  = (series.iloc[-1] / series.iloc[-MOMENTUM_PERIOD] - 1) * 100
        ret_5d   = (series.iloc[-1] / series.iloc[-5] - 1) * 100 if len(series) >= 6 else 0

        # Only include oversold stocks
        if rsi > RSI_THRESHOLD:
            continue

        rows.append({
            "ticker"    : ticker,
            "sector"    : SECTOR_ETFS.get(sector_etf, "Unknown") if sector_etf else "Unknown",
            "price"     : round(last_price, 2),
            "rsi"       : round(rsi, 1),
            "return_20d": round(float(ret_20d), 2),
            "return_5d" : round(float(ret_5d), 2),
            "avg_volume": int(avg_vol),
        })

    # Sort by RSI ascending (most oversold first)
    df = pd.DataFrame(rows).sort_values("rsi", ascending=True).head(TOP_N_STOCKS)
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    return df


# ===========================================================================
# SIGNAL 3: RELATIVE STRENGTH vs SPY
# ===========================================================================

def compute_relative_strength(
    prices    : pd.DataFrame,
    volumes   : pd.Series,
    top_sector_etfs: list,
) -> pd.DataFrame:
    """
    Ranks stocks by return vs SPY over 20 days.
    Positive = outperforming SPY, negative = underperforming.
    """
    if "SPY" not in prices.columns:
        log.warning("  SPY not in prices -- relative strength unavailable")
        return pd.DataFrame()

    spy_series = prices["SPY"].dropna()
    if len(spy_series) < RS_PERIOD + 1:
        return pd.DataFrame()

    spy_ret = float((spy_series.iloc[-1] / spy_series.iloc[-RS_PERIOD] - 1) * 100)

    rows = []
    for ticker in prices.columns:
        if ticker in SECTOR_ETFS or ticker == "SPY":
            continue

        sector_etf = SECTOR_MAP.get(ticker)
        if sector_etf and sector_etf not in top_sector_etfs:
            continue

        series = prices[ticker].dropna()
        if len(series) < RS_PERIOD + 1:
            continue

        last_price = float(series.iloc[-1])
        if last_price < MIN_PRICE:
            continue

        avg_vol = float(volumes.get(ticker, 0))
        if avg_vol < MIN_AVG_VOLUME:
            continue

        ret_20d = float((series.iloc[-1] / series.iloc[-RS_PERIOD] - 1) * 100)
        rs_vs_spy = ret_20d - spy_ret  # positive = beating SPY
        ret_5d  = float((series.iloc[-1] / series.iloc[-5] - 1) * 100) if len(series) >= 6 else 0

        rows.append({
            "ticker"      : ticker,
            "sector"      : SECTOR_ETFS.get(sector_etf, "Unknown") if sector_etf else "Unknown",
            "price"       : round(last_price, 2),
            "return_20d"  : round(ret_20d, 2),
            "vs_spy"      : round(rs_vs_spy, 2),
            "spy_ret_20d" : round(spy_ret, 2),
            "return_5d"   : round(ret_5d, 2),
            "avg_volume"  : int(avg_vol),
        })

    df = pd.DataFrame(rows).sort_values("vs_spy", ascending=False).head(TOP_N_STOCKS)
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    return df


# ===========================================================================
# MAIN
# ===========================================================================

def run_rotation():
    run_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    today  = datetime.today().strftime("%Y-%m-%d")

    log.info("=" * 60)
    log.info("  PORTFOLIO ROTATION STRATEGY")
    log.info(f"  Run: {run_ts}")
    log.info(f"  Top sectors    : {TOP_SECTORS}")
    log.info(f"  Stocks per list: {TOP_N_STOCKS}")
    log.info("=" * 60)

    # Build universe
    log.info("\n[1/5] Building universe ...")
    sp500    = fetch_sp500()
    all_tickers = list(dict.fromkeys(
        sp500 + list(SECTOR_ETFS.keys()) + ["SPY"]
    ))
    log.info(f"  Total tickers to download: {len(all_tickers)}")

    # Fetch prices and volumes
    log.info("\n[2/5] Fetching price data ...")
    prices  = fetch_prices(all_tickers)
    volumes = fetch_volumes(all_tickers)

    if prices.empty:
        log.error("No price data. Exiting.")
        return

    # Rank sectors
    log.info("\n[3/5] Ranking sectors ...")
    sector_ranks = rank_sectors(prices)
    top_sector_etfs = sector_ranks["etf"].head(TOP_SECTORS).tolist()

    log.info(f"  Sector rankings:")
    for _, row in sector_ranks.iterrows():
        marker = "  <-- TOP" if row["etf"] in top_sector_etfs else ""
        log.info(f"    {row.name}. {row['etf']:<5} {row['sector']:<25} {row['return_20d']:+.2f}%{marker}")

    # Compute the three lists
    log.info("\n[4/5] Computing signal lists ...")

    log.info("  Computing momentum ...")
    momentum_df = compute_momentum(prices, volumes, top_sector_etfs)
    log.info(f"  Momentum: {len(momentum_df)} stocks")

    log.info("  Computing mean reversion ...")
    reversion_df = compute_mean_reversion(prices, volumes, top_sector_etfs)
    log.info(f"  Mean reversion: {len(reversion_df)} stocks")

    log.info("  Computing relative strength ...")
    rs_df = compute_relative_strength(prices, volumes, top_sector_etfs)
    log.info(f"  Relative strength: {len(rs_df)} stocks")

    # Print to terminal
    print(f"\n{'='*60}")
    print(f"  MOMENTUM TOP {TOP_N_STOCKS} -- {today}")
    print(f"{'='*60}")
    if not momentum_df.empty:
        print(momentum_df[["ticker","sector","price","return_20d","return_5d"]].to_string())

    print(f"\n{'='*60}")
    print(f"  MEAN REVERSION TOP {TOP_N_STOCKS} (most oversold in strong sectors) -- {today}")
    print(f"{'='*60}")
    if not reversion_df.empty:
        print(reversion_df[["ticker","sector","price","rsi","return_20d"]].to_string())

    print(f"\n{'='*60}")
    print(f"  RELATIVE STRENGTH vs SPY TOP {TOP_N_STOCKS} -- {today}")
    print(f"{'='*60}")
    if not rs_df.empty:
        print(rs_df[["ticker","sector","price","vs_spy","return_20d"]].to_string())

    # Write to Google Sheets
    log.info("\n[5/5] Writing to Google Sheets ...")
    try:
        gc = _gs_client()
        ss = gc.open(GSHEET_NAME)

        # Sector tab
        sector_headers = ["run_date", "rank", "etf", "sector", "return_20d", "in_top5"]
        sector_rows = []
        for _, row in sector_ranks.iterrows():
            sector_rows.append([
                today,
                int(row.name),
                row["etf"],
                row["sector"],
                row["return_20d"],
                "YES" if row["etf"] in top_sector_etfs else "",
            ])
        write_ranked_list(ss, TAB_SECTORS, sector_headers, sector_rows, run_ts)

        # Momentum tab
        mom_headers = ["run_date", "rank", "ticker", "sector", "price", "return_20d_%", "return_5d_%", "avg_volume"]
        mom_rows = []
        for rank, row in momentum_df.iterrows():
            mom_rows.append([
                today, int(rank), row["ticker"], row["sector"],
                row["price"], row["return_20d"], row["return_5d"], row["avg_volume"],
            ])
        write_ranked_list(ss, TAB_MOMENTUM, mom_headers, mom_rows, run_ts)

        # Mean reversion tab
        rev_headers = ["run_date", "rank", "ticker", "sector", "price", "rsi", "return_20d_%", "return_5d_%", "avg_volume"]
        rev_rows = []
        for rank, row in reversion_df.iterrows():
            rev_rows.append([
                today, int(rank), row["ticker"], row["sector"],
                row["price"], row["rsi"], row["return_20d"], row["return_5d"], row["avg_volume"],
            ])
        write_ranked_list(ss, TAB_REVERSION, rev_headers, rev_rows, run_ts)

        # Relative strength tab
        rs_headers = ["run_date", "rank", "ticker", "sector", "price", "vs_spy_%", "return_20d_%", "spy_ret_20d_%", "return_5d_%", "avg_volume"]
        rs_rows = []
        for rank, row in rs_df.iterrows():
            rs_rows.append([
                today, int(rank), row["ticker"], row["sector"],
                row["price"], row["vs_spy"], row["return_20d"],
                row["spy_ret_20d"], row["return_5d"], row["avg_volume"],
            ])
        write_ranked_list(ss, TAB_RELSTRENGTH, rs_headers, rs_rows, run_ts)

        log.info("  All tabs written successfully")

    except Exception as e:
        log.error(f"  Google Sheets write failed: {e}")

    log.info("\n" + "=" * 60)
    log.info("  Rotation complete.")
    log.info("=" * 60)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    run_rotation()
