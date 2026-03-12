"""Scheduler-facing stock scrapping entrypoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import settings as cfg  # noqa: E402
import stock_scrapping as live_poller  # noqa: E402
from utils import kis_api  # noqa: E402


def _listify(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or list(default)
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or list(default)
    text = str(value).strip()
    return [text] if text else list(default)


def _resolve_path(value: Any, default: Path) -> Path:
    path = Path(value) if value else default
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_output_path() -> Path:
    configured_output = getattr(cfg, "STOCK_SCRAPPING_OUTPUT", None)
    if isinstance(configured_output, str) and re.fullmatch(r"live_ohlcv_\d+[hms]\.csv", configured_output):
        configured_output = live_poller.DEFAULT_OUTPUT
    return _resolve_path(configured_output, live_poller.DEFAULT_OUTPUT)


def _build_args() -> argparse.Namespace:
    return argparse.Namespace(
        stock_list=_resolve_path(getattr(cfg, "STOCK_SCRAPPING_STOCK_LIST", live_poller.DEFAULT_STOCK_LIST), live_poller.DEFAULT_STOCK_LIST),
        groups=_listify(getattr(cfg, "STOCK_SCRAPPING_GROUPS", ["kospi"]), ["kospi"]),
        symbols=_listify(getattr(cfg, "STOCK_SCRAPPING_SYMBOLS", []), []),
        limit=int(getattr(cfg, "STOCK_SCRAPPING_LIMIT", 0)),
        batch_size=int(getattr(cfg, "STOCK_SCRAPPING_BATCH_SIZE", live_poller.DEFAULT_BATCH_SIZE)),
        poll_interval=float(getattr(cfg, "STOCK_SCRAPPING_POLL_INTERVAL_SEC", live_poller.FIXED_BAR_INTERVAL_SEC)),
        min_request_gap=float(getattr(cfg, "STOCK_SCRAPPING_MIN_REQUEST_GAP_SEC", live_poller.DEFAULT_MIN_REQUEST_GAP_SEC)),
        max_retries=int(getattr(cfg, "STOCK_SCRAPPING_MAX_RETRIES", live_poller.DEFAULT_MAX_RETRIES)),
        cycles=0,
        output=_resolve_output_path(),
        exit_after_session=bool(getattr(cfg, "STOCK_SCRAPPING_EXIT_AFTER_SESSION", True)),
    )


def _tracked_markets(args: argparse.Namespace) -> set[str]:
    payload = json.loads(args.stock_list.read_text(encoding="utf-8"))
    groups = payload.get("groups", {})
    requested_groups = list(groups) if (not args.groups or "all" in args.groups) else list(args.groups)
    selected_symbols = {symbol.upper() for symbol in args.symbols}

    tracked: set[str] = set()
    for group_name in requested_groups:
        for item in groups.get(group_name, []):
            symbol = str(item.get("symbol") or "").strip().upper()
            if selected_symbols and symbol not in selected_symbols:
                continue
            venue = str(item.get("venue") or "").strip().upper()
            if venue == "KRX":
                tracked.add("KR")
            elif venue == "US":
                tracked.add("US")
    return tracked


def should_start_now(now: datetime | None = None) -> bool:
    args = _build_args()
    tracked_markets = _tracked_markets(args)
    if not tracked_markets:
        return False
    return any(live_poller.current_market_session(market_name, now) for market_name in tracked_markets)


def run(*, prefetched_token: str | None = None) -> dict[str, Any]:
    if prefetched_token:
        kis_api._token_cache["token"] = prefetched_token
        kis_api._token_cache["issued_at"] = time.time()

    args = _build_args()
    exit_code = live_poller.run(args)
    return {
        "status": "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "output": str(args.output),
        "groups": list(args.groups),
        "symbols": list(args.symbols),
        "exit_after_session": args.exit_after_session,
    }
