"""
Local-only derivatives scraper (KOSPI/KOSDAQ futures/options).

What this script does:
1) Polls KIS OpenAPI quote endpoints for configured derivative symbols.
2) Aggregates ticks into 1-minute OHLCV bars.
3) Computes futures basis (basis = futures_price - spot_index) when not provided.
4) Writes bars to a daily CSV and discards old-day files automatically.

Run:
    python quant_trading_system/SCRAP.py

Important:
- KIS futures/options TR IDs can differ by account mode/product.
- If requests fail with rt_cd/msg, replace API_PROFILE values using KIS docs.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

import settings as cfg
from utils.kis_api import APP_KEY, APP_SECRET, BASE_URL, get_access_token


KST = ZoneInfo("Asia/Seoul")
MARKET_OPEN = dt_time(9, 0)
MARKET_CLOSE = dt_time(15, 45)

POLL_INTERVAL_SEC = 2.0
REQUEST_GAP_SEC = 0.12
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "scrap"
OUTPUT_PREFIX = "derivatives_1m"

# Update these symbols to your target contracts (examples only).
DEFAULT_INSTRUMENTS = [
    {"symbol": "101V9000", "market": "KOSPI", "kind": "future"},
    {"symbol": "201V9000", "market": "KOSDAQ", "kind": "future"},
    {"symbol": "201W0000", "market": "KOSPI", "kind": "option"},
    {"symbol": "301W0000", "market": "KOSDAQ", "kind": "option"},
]


@dataclass(frozen=True)
class ApiProfile:
    path: str
    tr_id: str
    market_div: str
    price_keys: tuple[str, ...]
    basis_keys: tuple[str, ...]
    spot_keys: tuple[str, ...]
    volume_keys: tuple[str, ...]


# If your account returns API errors, replace these TR IDs/paths from KIS docs.
API_PROFILE: dict[str, ApiProfile] = {
    "future": ApiProfile(
        path="/uapi/domestic-futureoption/v1/quotations/inquire-price",
        tr_id="FHKIF03010100",
        market_div="F",
        price_keys=("futs_prpr", "stck_prpr", "prpr", "last", "cur_prc"),
        basis_keys=("futs_basi", "basis", "bstp_nmix_prpr"),
        spot_keys=("spot_prpr", "idx_prpr", "bstp_nmix_prpr"),
        volume_keys=("acml_vol", "cvolume", "tday_rltv", "cntg_vol"),
    ),
    "option": ApiProfile(
        path="/uapi/domestic-futureoption/v1/quotations/inquire-price",
        tr_id="FHKIF03010200",
        market_div="O",
        price_keys=("optn_prpr", "stck_prpr", "prpr", "last", "cur_prc"),
        basis_keys=("",),
        spot_keys=("",),
        volume_keys=("acml_vol", "cvolume", "tday_rltv", "cntg_vol"),
    ),
}


@dataclass
class MinuteBar:
    minute: datetime
    symbol: str
    market: str
    kind: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    basis: float | None = None
    spot_index: float | None = None
    fut_price: float | None = None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def as_float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def first_num(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if not key:
            continue
        num = as_float(payload.get(key))
        if num is not None:
            return num
    return None


def market_open_now(now_kst: datetime) -> bool:
    if now_kst.weekday() >= 5:
        return False
    return MARKET_OPEN <= now_kst.time() <= MARKET_CLOSE


def floor_minute(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)


def daily_csv_path(day_str: str) -> Path:
    return OUTPUT_DIR / f"{OUTPUT_PREFIX}_{day_str}.csv"


def purge_other_days(today_str: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.glob(f"{OUTPUT_PREFIX}_*.csv"):
        if today_str not in path.name:
            path.unlink(missing_ok=True)


def csv_headers() -> list[str]:
    return [
        "date",
        "minute",
        "symbol",
        "market",
        "kind",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "basis",
        "spot_index",
        "fut_price",
        "collected_at",
    ]


def write_bars(path: Path, bars: list[MinuteBar], collected_at: datetime) -> None:
    if not bars:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if write_header:
            writer.writerow(csv_headers())
        for bar in sorted(bars, key=lambda x: (x.minute, x.symbol, x.kind)):
            writer.writerow(
                [
                    bar.minute.strftime("%Y-%m-%d"),
                    bar.minute.strftime("%H:%M:%S"),
                    bar.symbol,
                    bar.market,
                    bar.kind,
                    f"{bar.open:.6f}",
                    f"{bar.high:.6f}",
                    f"{bar.low:.6f}",
                    f"{bar.close:.6f}",
                    f"{bar.volume:.6f}",
                    "" if bar.basis is None else f"{bar.basis:.6f}",
                    "" if bar.spot_index is None else f"{bar.spot_index:.6f}",
                    "" if bar.fut_price is None else f"{bar.fut_price:.6f}",
                    collected_at.isoformat(),
                ]
            )


def build_headers(access_token: str, tr_id: str) -> dict[str, str]:
    return {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {access_token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": tr_id,
    }


def fetch_tick(
    session: requests.Session,
    instrument: dict[str, str],
) -> dict[str, Any] | None:
    kind = instrument["kind"]
    profile = API_PROFILE[kind]
    token = get_access_token()
    headers = build_headers(token, profile.tr_id)
    params = {
        "fid_cond_mrkt_div_code": profile.market_div,
        "fid_input_iscd": instrument["symbol"],
    }
    url = f"{BASE_URL}{profile.path}"

    try:
        res = session.get(url, headers=headers, params=params, timeout=cfg.API_TIMEOUT_SEC)
        data = res.json()
    except Exception as exc:
        logging.warning("Request failed for %s (%s): %s", instrument["symbol"], kind, exc)
        return None

    if data.get("rt_cd") != "0":
        logging.warning(
            "KIS rejected %s (%s): rt_cd=%s msg=%s",
            instrument["symbol"],
            kind,
            data.get("rt_cd"),
            data.get("msg1", ""),
        )
        return None

    output = data.get("output") or {}
    price = first_num(output, profile.price_keys)
    if price is None:
        return None

    basis = first_num(output, profile.basis_keys)
    spot = first_num(output, profile.spot_keys)
    if kind == "future" and basis is None and spot is not None:
        basis = price - spot

    volume = first_num(output, profile.volume_keys)
    now_kst = datetime.now(KST)

    return {
        "timestamp": now_kst,
        "symbol": instrument["symbol"],
        "market": instrument["market"],
        "kind": kind,
        "price": price,
        "volume": volume,
        "basis": basis,
        "spot": spot,
    }


def run() -> None:
    setup_logging()
    instruments = list(DEFAULT_INSTRUMENTS)
    if not instruments:
        raise RuntimeError("No instruments configured in DEFAULT_INSTRUMENTS")

    logging.info("SCRAP started (local mode). symbols=%d", len(instruments))
    bars: dict[tuple[datetime, str, str], MinuteBar] = {}
    last_cum_vol: dict[str, float] = {}

    now = datetime.now(KST)
    day_key = now.strftime("%Y%m%d")
    purge_other_days(day_key)

    session = requests.Session()
    out_path = daily_csv_path(day_key)

    try:
        while True:
            now = datetime.now(KST)
            new_day = now.strftime("%Y%m%d")
            if new_day != day_key:
                write_bars(out_path, list(bars.values()), now)
                bars.clear()
                last_cum_vol.clear()
                day_key = new_day
                out_path = daily_csv_path(day_key)
                purge_other_days(day_key)
                logging.info("Day rollover: old data discarded, new file=%s", out_path.name)

            if not market_open_now(now):
                flush_minute = floor_minute(now)
                done_keys = [key for key in bars if key[0] < flush_minute]
                if done_keys:
                    done = [bars.pop(key) for key in done_keys]
                    write_bars(out_path, done, now)
                time.sleep(5.0)
                continue

            for inst in instruments:
                tick = fetch_tick(session, inst)
                if tick is None:
                    time.sleep(REQUEST_GAP_SEC)
                    continue

                minute_key = floor_minute(tick["timestamp"])
                key = (minute_key, tick["symbol"], tick["kind"])
                bar = bars.get(key)
                price = tick["price"]
                if bar is None:
                    bars[key] = MinuteBar(
                        minute=minute_key,
                        symbol=tick["symbol"],
                        market=tick["market"],
                        kind=tick["kind"],
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=0.0,
                        basis=tick["basis"],
                        spot_index=tick["spot"],
                        fut_price=price if tick["kind"] == "future" else None,
                    )
                else:
                    bar.high = max(bar.high, price)
                    bar.low = min(bar.low, price)
                    bar.close = price
                    if tick["basis"] is not None:
                        bar.basis = tick["basis"]
                    if tick["spot"] is not None:
                        bar.spot_index = tick["spot"]
                    if tick["kind"] == "future":
                        bar.fut_price = price

                if tick["volume"] is not None:
                    prev = last_cum_vol.get(tick["symbol"])
                    if prev is None or tick["volume"] < prev:
                        vol_delta = 0.0
                    else:
                        vol_delta = tick["volume"] - prev
                    last_cum_vol[tick["symbol"]] = tick["volume"]
                    bars[key].volume += max(vol_delta, 0.0)

                time.sleep(REQUEST_GAP_SEC)

            flush_before = floor_minute(datetime.now(KST))
            done_keys = [key for key in bars if key[0] < flush_before]
            if done_keys:
                done = [bars.pop(key) for key in done_keys]
                write_bars(out_path, done, datetime.now(KST))
                logging.info("Flushed %d minute bars -> %s", len(done), out_path.name)

            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        logging.info("Stopped by user (Ctrl+C)")
        if bars:
            write_bars(out_path, list(bars.values()), datetime.now(KST))
            bars.clear()


if __name__ == "__main__":
    run()
#python quant_trading_system/SCRAP.py