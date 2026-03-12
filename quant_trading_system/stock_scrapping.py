from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
QUANT_SYSTEM_ROOT = PROJECT_ROOT / "quant_trading_system"
if str(QUANT_SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_SYSTEM_ROOT))

import settings as cfg  # noqa: E402
from utils import kis_api  # noqa: E402

LOGGER = logging.getLogger("live_ohlcv_poller")
KST = ZoneInfo("Asia/Seoul")
US_ET = ZoneInfo("America/New_York")
DEFAULT_STOCK_LIST = PROJECT_ROOT / "stock_list.json"
FIXED_BAR_INTERVAL_SEC = 300
DEFAULT_POLL_INTERVAL_SEC = float(os.getenv("KIS_POLL_INTERVAL_SEC", str(FIXED_BAR_INTERVAL_SEC)))
DEFAULT_MIN_REQUEST_GAP_SEC = float(os.getenv("KIS_MIN_REQUEST_GAP_SEC", "1"))
DEFAULT_BATCH_SIZE = int(os.getenv("KIS_BATCH_SIZE", "250"))
DEFAULT_MAX_RETRIES = int(os.getenv("KIS_MAX_RETRIES", "5"))
DEFAULT_MAX_CYCLES = int(os.getenv("KIS_MAX_CYCLES", "0"))
POLL_ALIGN_DELAY_SEC = float(os.getenv("KIS_POLL_ALIGN_DELAY_SEC", "0"))
DOMESTIC_INTRADAY_PATH = "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
DOMESTIC_INTRADAY_TR_ID = "FHKST03010200"
OVERSEAS_INTRADAY_PATH = "/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice"
OVERSEAS_INTRADAY_TR_ID = "HHDFS76950200"
LIMIT_HINTS = ("초당", "유량", "호출", "too many", "rate limit", "limit exceeded")


def _interval_suffix(interval_sec: int) -> str:
    if interval_sec % 3600 == 0:
        return f"{interval_sec // 3600}h"
    if interval_sec % 60 == 0:
        return f"{interval_sec // 60}m"
    return f"{interval_sec}s"


def _interval_display(interval_sec: int) -> str:
    if interval_sec % 3600 == 0:
        hours = interval_sec // 3600
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    if interval_sec % 60 == 0:
        minutes = interval_sec // 60
        return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"
    return f"{interval_sec} seconds"


DEFAULT_OUTPUT = PROJECT_ROOT / f"live_ohlcv_{_interval_suffix(FIXED_BAR_INTERVAL_SEC)}.csv"


@dataclass(frozen=True)
class StockSpec:
    symbol: str
    venue: str
    group: str
    market: str
    request_symbol: str
    market_div_code: str = "J"
    exchange_code: str = ""
    display_name: str = ""


@dataclass(frozen=True)
class MarketWindow:
    market_name: str
    tz: ZoneInfo
    opens_at: dt_time
    closes_at: dt_time


@dataclass(frozen=True)
class MarketSession:
    market_name: str
    session_key: str
    started_at: datetime
    ends_at: datetime


MARKET_WINDOWS = {
    "KR": MarketWindow("KR", KST, dt_time(9, 0), dt_time(15, 30)),
    "US": MarketWindow("US", US_ET, dt_time(9, 30), dt_time(16, 0)),
}


class RequestPacer:
    def __init__(self, min_interval_sec: float) -> None:
        self.min_interval_sec = max(0.0, min_interval_sec)
        self._last_request_monotonic = 0.0
        self._cooldown_until = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        target = max(self._cooldown_until, self._last_request_monotonic + self.min_interval_sec)
        sleep_for = target - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last_request_monotonic = time.monotonic()

    def backoff(self, attempt: int) -> float:
        delay = min(30.0, 1.5 * (2 ** attempt) + random.uniform(0.0, 0.35))
        self._cooldown_until = max(self._cooldown_until, time.monotonic() + delay)
        return delay


class KisRequestError(RuntimeError):
    pass


class KisRateLimitError(KisRequestError):
    pass


class SessionTokenManager:
    """Acquire at most one token per active market session and discard it at session close."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._session: MarketSession | None = None

    def _clear_underlying_cache(self) -> None:
        if hasattr(kis_api, "_token_cache"):
            kis_api._token_cache["token"] = None
            kis_api._token_cache["issued_at"] = 0

    def discard(self, reason: str) -> None:
        if self._session or self._token:
            LOGGER.info("Discarding KIS token for session=%s (%s)", self._session.session_key if self._session else "none", reason)
        self._token = None
        self._session = None

    def current_session(self) -> MarketSession | None:
        return self._session

    def ensure_token(self, market_name: str, now_utc: datetime | None = None) -> str:
        session = current_market_session(market_name, now_utc)
        if session is None:
            if self._session and self._session.market_name == market_name:
                self.discard(f"{market_name} market closed")
            raise RuntimeError(f"{market_name} market is closed")

        if self._session is None or self._session.session_key != session.session_key:
            self.discard(f"new {market_name} session")
            self._token = kis_api.get_access_token()
            self._session = session
            LOGGER.info(
                "Issued one KIS token for %s session %s (valid until market close %s)",
                market_name,
                session.session_key,
                session.ends_at.isoformat(),
            )

        return self._token

    def discard_if_outside_sessions(self, tracked_markets: set[str], now_utc: datetime | None = None) -> None:
        if not self._session:
            return
        now_utc = now_utc or datetime.now(timezone.utc)
        active_session = current_market_session(self._session.market_name, now_utc)
        if self._session.market_name not in tracked_markets or active_session is None or active_session.session_key != self._session.session_key:
            self.discard("session no longer active")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-7s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _csv_list(value: str | None, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


def _to_float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in (None, "", "-"):
        return None
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return None


def _kis_headers(token: str, tr_id: str) -> dict[str, str]:
    return {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": kis_api.APP_KEY,
        "appsecret": kis_api.APP_SECRET,
        "tr_id": tr_id,
    }


def market_name_for_spec(spec: StockSpec) -> str:
    return "KR" if spec.venue == "KRX" else "US"


def current_market_session(market_name: str, now_utc: datetime | None = None) -> MarketSession | None:
    window = MARKET_WINDOWS[market_name]
    now_utc = now_utc or datetime.now(timezone.utc)
    local_now = now_utc.astimezone(window.tz)
    if local_now.weekday() >= 5:
        return None
    if not (window.opens_at <= local_now.time() <= window.closes_at):
        return None

    started_at = datetime.combine(local_now.date(), window.opens_at, tzinfo=window.tz)
    ends_at = datetime.combine(local_now.date(), window.closes_at, tzinfo=window.tz)
    return MarketSession(
        market_name=market_name,
        session_key=f"{market_name}:{local_now.date().isoformat()}",
        started_at=started_at.astimezone(timezone.utc),
        ends_at=ends_at.astimezone(timezone.utc),
    )


def next_market_open(markets: set[str], now_utc: datetime | None = None) -> datetime:
    now_utc = now_utc or datetime.now(timezone.utc)
    candidates: list[datetime] = []
    for market_name in markets:
        window = MARKET_WINDOWS[market_name]
        local_now = now_utc.astimezone(window.tz)
        for offset in range(8):
            candidate_date = local_now.date() + timedelta(days=offset)
            if candidate_date.weekday() >= 5:
                continue
            candidate_local = datetime.combine(candidate_date, window.opens_at, tzinfo=window.tz)
            if candidate_local.astimezone(timezone.utc) > now_utc:
                candidates.append(candidate_local.astimezone(timezone.utc))
                break
    if not candidates:
        return now_utc + timedelta(minutes=30)
    return min(candidates)


def floor_time(ts: datetime, interval_sec: int) -> datetime:
    base = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = int((ts - base).total_seconds())
    floored = elapsed - (elapsed % interval_sec)
    return base + timedelta(seconds=floored)


def next_aligned_poll_time(now_utc: datetime, interval_sec: int, market_tz: ZoneInfo) -> datetime:
    local_now = now_utc.astimezone(market_tz)
    base = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = int((local_now - base).total_seconds())
    next_elapsed = ((elapsed // interval_sec) + 1) * interval_sec
    target_local = base + timedelta(seconds=next_elapsed + POLL_ALIGN_DELAY_SEC)
    return target_local.astimezone(timezone.utc)


def is_aligned_poll_time(now_utc: datetime, interval_sec: int, market_tz: ZoneInfo) -> bool:
    local_now = now_utc.astimezone(market_tz)
    base = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = int((local_now - base).total_seconds())
    delay = float(POLL_ALIGN_DELAY_SEC)
    if elapsed < delay:
        return False
    return ((elapsed - delay) % interval_sec) == 0


def first_completed_bar_time(market_name: str, now_utc: datetime | None = None, interval_sec: int = FIXED_BAR_INTERVAL_SEC) -> datetime | None:
    window = MARKET_WINDOWS[market_name]
    now_utc = now_utc or datetime.now(timezone.utc)
    local_now = now_utc.astimezone(window.tz)
    if local_now.weekday() >= 5:
        return None
    open_local = datetime.combine(local_now.date(), window.opens_at, tzinfo=window.tz)
    return (open_local + timedelta(seconds=interval_sec)).astimezone(timezone.utc)


def has_completed_bar_available(market_name: str, now_utc: datetime | None = None, interval_sec: int = FIXED_BAR_INTERVAL_SEC) -> bool:
    first_ready = first_completed_bar_time(market_name, now_utc, interval_sec)
    if first_ready is None:
        return False
    now_utc = now_utc or datetime.now(timezone.utc)
    return now_utc >= first_ready


def next_ready_poll_time(market_name: str, now_utc: datetime | None = None, interval_sec: int = FIXED_BAR_INTERVAL_SEC) -> datetime:
    now_utc = now_utc or datetime.now(timezone.utc)
    first_ready = first_completed_bar_time(market_name, now_utc, interval_sec)
    if first_ready and now_utc < first_ready:
        return first_ready
    return next_aligned_poll_time(now_utc, interval_sec, MARKET_WINDOWS[market_name].tz)


def _load_stock_specs(stock_list_path: Path, groups: list[str], limit: int, symbols: set[str]) -> list[StockSpec]:
    payload = json.loads(stock_list_path.read_text(encoding="utf-8"))
    group_map: dict[str, list[dict[str, Any]]] = payload.get("groups", {})
    requested_groups = list(group_map) if (not groups or "all" in groups) else groups

    specs: list[StockSpec] = []
    seen: set[str] = set()

    for group_name in requested_groups:
        items = group_map.get(group_name, [])
        if not items:
            LOGGER.warning("group '%s' not found in %s", group_name, stock_list_path)
            continue

        for item in items:
            symbol = str(item.get("symbol") or "").strip().upper()
            venue = str(item.get("venue", "")).strip().upper()
            if not symbol or symbol in seen:
                continue
            if venue not in {"KRX", "US"}:
                continue
            if symbols and symbol not in symbols:
                continue

            seen.add(symbol)
            request_symbol = str(item.get("request_symbol") or symbol).strip().upper()
            exchange_code = str(item.get("exchange_code", "")).strip().upper()
            specs.append(
                StockSpec(
                    symbol=symbol,
                    venue=venue,
                    group=str(item.get("group", group_name)),
                    market=str(item.get("market", group_name)).strip(),
                    request_symbol=request_symbol,
                    market_div_code=str(item.get("market_div_code", "J") or "J"),
                    exchange_code=exchange_code,
                    display_name=str(item.get("name_kr") or item.get("name_en") or symbol).strip(),
                )
            )

    if limit > 0:
        specs = specs[:limit]

    if symbols:
        found = {spec.symbol for spec in specs}
        missing = sorted(symbols - found)
        if missing:
            LOGGER.warning("symbols not found/unsupported in stock_list.json: %s", ", ".join(missing))

    return specs


def _is_rate_limited(payload: dict[str, Any], status_code: int) -> bool:
    if status_code == 429:
        return True
    message = f"{payload.get('msg_cd', '')} {payload.get('msg1', '')}".lower()
    return any(hint in message for hint in LIMIT_HINTS)


def _request_json(
    session: requests.Session,
    pacer: RequestPacer,
    token_manager: SessionTokenManager,
    market_name: str,
    endpoint: str,
    tr_id: str,
    params: dict[str, Any],
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            token = token_manager.ensure_token(market_name)
            pacer.wait()
            response = session.get(
                f"{kis_api.BASE_URL}{endpoint}",
                headers=_kis_headers(token, tr_id),
                params=params,
                timeout=cfg.API_TIMEOUT_SEC,
            )
            payload = response.json()
            if response.status_code == 401:
                raise KisRequestError("session token expired or rejected; configured policy forbids reissuing within the same market session")
            if _is_rate_limited(payload, response.status_code):
                raise KisRateLimitError(payload.get("msg1", "rate limit"))
            if response.status_code >= 400:
                response.raise_for_status()
            if str(payload.get("rt_cd", "1")) != "0":
                raise KisRequestError(payload.get("msg1") or str(payload))
            return payload
        except (requests.RequestException, ValueError, KisRequestError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            delay = pacer.backoff(attempt) if isinstance(exc, KisRateLimitError) else min(10.0, 0.8 * (2 ** attempt))
            LOGGER.warning(
                "request retry %s/%s for %s %s after error: %s (sleep %.2fs)",
                attempt + 1,
                max_retries,
                tr_id,
                params.get("FID_INPUT_ISCD") or params.get("SYMB", ""),
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(f"KIS request failed for {tr_id} {params}: {last_error}")


def parse_kr_row_timestamp(item: dict[str, Any]) -> datetime:
    return datetime.strptime(
        f"{str(item.get('stck_bsop_date')).strip()}{str(item.get('stck_cntg_hour')).strip().zfill(6)}",
        "%Y%m%d%H%M%S",
    ).replace(tzinfo=KST)


def parse_us_row_timestamp(item: dict[str, Any]) -> datetime:
    if item.get("kymd") and item.get("khms"):
        return datetime.strptime(
            f"{str(item.get('kymd')).strip()}{str(item.get('khms')).strip().zfill(6)}",
            "%Y%m%d%H%M%S",
        ).replace(tzinfo=KST)
    return datetime.strptime(
        f"{str(item.get('xymd')).strip()}{str(item.get('xhms')).strip().zfill(6)}",
        "%Y%m%d%H%M%S",
    ).replace(tzinfo=US_ET).astimezone(KST)


def aggregate_kr_bar(spec: StockSpec, rows: list[dict[str, Any]], interval_sec: int, polled_at: datetime) -> dict[str, Any]:
    cutoff = floor_time(polled_at.astimezone(KST), interval_sec)
    grouped: dict[datetime, list[tuple[datetime, dict[str, Any]]]] = {}
    for item in rows:
        try:
            ts = parse_kr_row_timestamp(item)
        except Exception:
            continue
        bucket_start = floor_time(ts, interval_sec)
        bucket_end = bucket_start + timedelta(seconds=interval_sec)
        if bucket_end <= cutoff:
            grouped.setdefault(bucket_start, []).append((ts, item))
    if not grouped:
        raise RuntimeError(f"No completed KR {_interval_display(interval_sec)} bucket available")
    bucket_start = max(grouped)
    bucket_end = bucket_start + timedelta(seconds=interval_sec)
    bucket_rows = sorted(grouped[bucket_start], key=lambda x: x[0])
    opens = [_to_float(item.get("stck_oprc")) for _, item in bucket_rows]
    highs = [_to_float(item.get("stck_hgpr")) for _, item in bucket_rows]
    lows = [_to_float(item.get("stck_lwpr")) for _, item in bucket_rows]
    closes = [_to_float(item.get("stck_prpr") or item.get("stck_clpr")) for _, item in bucket_rows]
    vols = [_to_int(item.get("cntg_vol") or 0) or 0 for _, item in bucket_rows]
    return {
        "polled_at_utc": polled_at.astimezone(timezone.utc).isoformat(),
        "bar_start": bucket_start.isoformat(),
        "bar_end": bucket_end.isoformat(),
        "market_timestamp": bucket_start.isoformat(),
        "bar_interval_sec": interval_sec,
        "is_partial_bar": False,
        "symbol": spec.symbol,
        "venue": spec.venue,
        "group": spec.group,
        "market": spec.market,
        "name": spec.display_name,
        "open": opens[0],
        "high": max(v for v in highs if v is not None),
        "low": min(v for v in lows if v is not None),
        "close": closes[-1],
        "volume": sum(vols),
        "source_endpoint": DOMESTIC_INTRADAY_PATH,
        "tr_id": DOMESTIC_INTRADAY_TR_ID,
    }


def normalize_us_bar(spec: StockSpec, item: dict[str, Any], interval_sec: int, polled_at: datetime) -> dict[str, Any]:
    row_ts = parse_us_row_timestamp(item)
    bucket_start = floor_time(row_ts, interval_sec)
    bucket_end = bucket_start + timedelta(seconds=interval_sec)
    return {
        "polled_at_utc": polled_at.astimezone(timezone.utc).isoformat(),
        "bar_start": bucket_start.isoformat(),
        "bar_end": bucket_end.isoformat(),
        "market_timestamp": bucket_start.isoformat(),
        "bar_interval_sec": interval_sec,
        "is_partial_bar": False,
        "symbol": spec.symbol,
        "venue": spec.venue,
        "group": spec.group,
        "market": spec.market,
        "name": spec.display_name,
        "open": _to_float(item.get("open")),
        "high": _to_float(item.get("high")),
        "low": _to_float(item.get("low")),
        "close": _to_float(item.get("last")),
        "volume": _to_int(item.get("evol") or item.get("tvol") or item.get("pvol")),
        "source_endpoint": OVERSEAS_INTRADAY_PATH,
        "tr_id": OVERSEAS_INTRADAY_TR_ID,
    }


def select_latest_completed_us_bar(rows: list[dict[str, Any]], interval_sec: int, polled_at: datetime) -> dict[str, Any]:
    cutoff = floor_time(polled_at.astimezone(US_ET), interval_sec)
    candidates: list[tuple[datetime, dict[str, Any]]] = []
    for item in rows:
        try:
            ts = parse_us_row_timestamp(item).astimezone(US_ET)
        except Exception:
            continue
        if ts < cutoff:
            candidates.append((ts, item))
    if not candidates:
        raise RuntimeError(f"No completed US {_interval_display(interval_sec)} bucket available")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _fetch_live_ohlcv(
    session: requests.Session,
    pacer: RequestPacer,
    token_manager: SessionTokenManager,
    spec: StockSpec,
    max_retries: int,
    interval_sec: int,
) -> dict[str, Any]:
    polled_at = datetime.now(timezone.utc)

    if spec.venue == "KRX":
        payload = _request_json(
            session=session,
            pacer=pacer,
            token_manager=token_manager,
            market_name=market_name_for_spec(spec),
            endpoint=DOMESTIC_INTRADAY_PATH,
            tr_id=DOMESTIC_INTRADAY_TR_ID,
            params={
                "FID_ETC_CLS_CODE": "",
                "FID_COND_MRKT_DIV_CODE": spec.market_div_code,
                "FID_INPUT_ISCD": spec.symbol,
                "FID_INPUT_HOUR_1": datetime.now(KST).strftime("%H%M%S"),
                "FID_PW_DATA_INCU_YN": "Y",
            },
            max_retries=max_retries,
        )
        rows = payload.get("output2") or []
        return aggregate_kr_bar(spec, rows, interval_sec, polled_at)

    if spec.venue == "US":
        payload = _request_json(
            session=session,
            pacer=pacer,
            token_manager=token_manager,
            market_name=market_name_for_spec(spec),
            endpoint=OVERSEAS_INTRADAY_PATH,
            tr_id=OVERSEAS_INTRADAY_TR_ID,
            params={
                "AUTH": "",
                "EXCD": spec.exchange_code,
                "SYMB": spec.request_symbol.split(":", 1)[-1],
                "NMIN": str(max(1, interval_sec // 60)),
                "PINC": "1",
                "NEXT": "",
                "NREC": "120",
                "FILL": "",
                "KEYB": "",
            },
            max_retries=max_retries,
        )
        rows = payload.get("output2") or []
        item = select_latest_completed_us_bar(rows, interval_sec, polled_at)
        return normalize_us_bar(spec, item, interval_sec, polled_at)

    raise ValueError(f"Unsupported venue: {spec.venue}")


def _append_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    fieldnames = list(rows[0].keys())
    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _print_preview(rows: list[dict[str, Any]], limit: int = 10) -> None:
    if not rows:
        return
    fields = ["symbol", "open", "high", "low", "close", "volume"]
    preview = rows[-min(limit, len(rows)):]
    print(" ".join(f"{f:>12}" for f in fields))
    for row in preview:
        print(" ".join(f"{str(row.get(f, '')):>12}" for f in fields))


def _safe_symbol_capacity(poll_interval: float, min_request_gap: float, batch_size: int) -> int:
    by_gap = max(1, int(poll_interval // max(min_request_gap, 0.01)))
    return max(1, min(batch_size, by_gap))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Live {_interval_display(FIXED_BAR_INTERVAL_SEC)} OHLCV poller using quant_trading_system/utils/kis_api.py"
    )
    parser.add_argument("--stock-list", type=Path, default=DEFAULT_STOCK_LIST)
    parser.add_argument("--groups", nargs="+", default=_csv_list(os.getenv("KIS_GROUPS"), ["all"]))
    parser.add_argument("--symbols", nargs="*", default=_csv_list(os.getenv("KIS_SYMBOLS"), []))
    parser.add_argument("--limit", type=int, default=int(os.getenv("KIS_LIMIT", "0")), help="0 means auto-safe cap")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SEC,
        help=f"ignored; bar mode uses a fixed {_interval_display(FIXED_BAR_INTERVAL_SEC)} interval ({FIXED_BAR_INTERVAL_SEC} seconds)",
    )
    parser.add_argument("--min-request-gap", type=float, default=DEFAULT_MIN_REQUEST_GAP_SEC)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--cycles", type=int, default=DEFAULT_MAX_CYCLES, help="0 means run forever")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--exit-after-session",
        action="store_true",
        help="exit once the tracked market session that this process observed has closed",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if not kis_api.APP_KEY or not kis_api.APP_SECRET:
        raise ValueError("quant_trading_system/settings.py must contain KIS_APP_KEY and KIS_APP_SECRET")

    interval_sec = FIXED_BAR_INTERVAL_SEC
    if int(args.poll_interval) != FIXED_BAR_INTERVAL_SEC:
        LOGGER.warning(
            "Ignoring requested poll interval %s; bar mode uses a fixed %s interval (%s seconds).",
            args.poll_interval,
            _interval_display(interval_sec),
            interval_sec,
        )
    safe_capacity = _safe_symbol_capacity(interval_sec, args.min_request_gap, args.batch_size)
    requested_limit = args.limit if args.limit > 0 else safe_capacity
    effective_limit = min(requested_limit, safe_capacity)

    if requested_limit > safe_capacity:
        LOGGER.warning(
            "Requested symbol count %s exceeds safe %s capacity %s. Truncating to %s to avoid KIS rate limit and preserve the configured %s cadence.",
            requested_limit,
            _interval_display(interval_sec),
            safe_capacity,
            effective_limit,
            _interval_display(interval_sec),
        )

    symbols = {symbol.upper() for symbol in args.symbols}
    specs = _load_stock_specs(args.stock_list, args.groups, effective_limit, symbols)
    if not specs:
        raise ValueError("No supported KRX/US symbols resolved from stock_list.json with the given filters")

    tracked_markets = {market_name_for_spec(spec) for spec in specs}
    pacer = RequestPacer(args.min_request_gap)
    token_manager = SessionTokenManager()
    session = requests.Session()
    cycle = 0
    seen_active_session = False

    LOGGER.info(
        "starting session-scoped %s poller | selected_symbols=%s | tracked_markets=%s | safe_capacity=%s | poll_interval=%ss",
        _interval_display(interval_sec),
        len(specs),
        sorted(tracked_markets),
        safe_capacity,
        interval_sec,
    )

    while True:
        now_utc = datetime.now(timezone.utc)
        token_manager.discard_if_outside_sessions(tracked_markets, now_utc)
        session_specs = [spec for spec in specs if current_market_session(market_name_for_spec(spec), now_utc)]
        if session_specs:
            seen_active_session = True
        elif args.exit_after_session and seen_active_session:
            token_manager.discard("tracked market session finished")
            LOGGER.info("Tracked market session closed after active polling window. Exiting.")
            return 0
        active_specs = [spec for spec in session_specs if has_completed_bar_available(market_name_for_spec(spec), now_utc, interval_sec)]

        if active_specs:
            aligned_targets = {market_name_for_spec(spec): MARKET_WINDOWS[market_name_for_spec(spec)].tz for spec in active_specs}
            if not any(is_aligned_poll_time(now_utc, interval_sec, tz) for tz in aligned_targets.values()):
                next_poll_utc = min(next_ready_poll_time(market, now_utc, interval_sec) for market in aligned_targets)
                sleep_for = max(0.0, (next_poll_utc - now_utc).total_seconds())
                if sleep_for > 0:
                    LOGGER.info(
                        "Waiting %.0fs for next aligned %s boundary at %s",
                        sleep_for,
                        _interval_display(interval_sec),
                        next_poll_utc.isoformat(),
                    )
                    time.sleep(sleep_for)
                continue

        if not active_specs:
            if session_specs:
                next_poll_utc = min(next_ready_poll_time(market_name_for_spec(spec), now_utc, interval_sec) for spec in session_specs)
                sleep_for = max(1.0, min(900.0, (next_poll_utc - now_utc).total_seconds()))
                LOGGER.info(
                    "Market is open but no completed %s bar is ready yet. Sleeping %.0fs until %s.",
                    _interval_display(interval_sec),
                    sleep_for,
                    next_poll_utc.isoformat(),
                )
            else:
                next_open = next_market_open(tracked_markets, now_utc)
                next_poll_utc = next_open + timedelta(seconds=interval_sec)
                sleep_for = max(1.0, min(900.0, (next_poll_utc - now_utc).total_seconds()))
                LOGGER.info(
                    "No active tracked market session. Token discarded if needed. Sleeping %.0fs until next completed-bar window (%s).",
                    sleep_for,
                    next_poll_utc.isoformat(),
                )
            time.sleep(sleep_for)
            if args.cycles and cycle >= args.cycles:
                return 0
            continue

        cycle += 1
        cycle_started = time.monotonic()
        rows: list[dict[str, Any]] = []
        failures: list[str] = []

        for spec in active_specs:
            try:
                rows.append(_fetch_live_ohlcv(session, pacer, token_manager, spec, args.max_retries, interval_sec))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{spec.symbol}: {exc}")
                LOGGER.error("live OHLCV fetch failed for %s: %s", spec.symbol, exc)

        _append_rows(args.output, rows)
        _print_preview(rows)
        LOGGER.info(
            "cycle=%s complete | active_symbols=%s | success=%s | failed=%s | session=%s",
            cycle,
            len(active_specs),
            len(rows),
            len(failures),
            token_manager.current_session().session_key if token_manager.current_session() else "none",
        )
        if failures:
            LOGGER.warning("cycle=%s failures: %s", cycle, " | ".join(failures[:5]))

        if args.cycles and cycle >= args.cycles:
            token_manager.discard("cycle limit reached")
            return 0

        next_targets = [next_aligned_poll_time(datetime.now(timezone.utc), interval_sec, MARKET_WINDOWS[m].tz) for m in tracked_markets]
        next_poll_utc = min(next_targets)
        sleep_for = max(0.0, (next_poll_utc - datetime.now(timezone.utc)).total_seconds())
        if sleep_for > 0:
            time.sleep(sleep_for)


def main() -> int:
    _setup_logging()
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        LOGGER.info("live poller interrupted by user")
        return 130
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("live poller failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
