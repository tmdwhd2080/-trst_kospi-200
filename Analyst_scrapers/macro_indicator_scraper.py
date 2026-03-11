"""
매크로 지표 수집기
- VIX (CBOE 변동성 지수)
- MOVE Index (채권 변동성)
- 한국 CDS 5Y 프리미엄
- 원/달러 환율
"""

# ═══════════════════════════════════════════════════
#  ⚙ 수집기 개별 설정 — 이 영역의 값만 수정하세요
# ═══════════════════════════════════════════════════
VIX_THRESHOLDS = {
    "extreme_greed": 12,    # 이하 → Extreme Greed
    "greed": 18,            # 이하 → Greed
    "neutral": 25,          # 이하 → Neutral
    "fear": 30,             # 이하 → Fear
    # 초과 → Extreme Fear
}
# ═══════════════════════════════════════════════════

import logging
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _safe_float(text: str) -> float | None:
    """문자열 → float 변환. 실패 시 None."""
    try:
        return float(text.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def fetch_vix() -> float | None:
    """Yahoo Finance에서 VIX 지수 조회."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
    params = {"interval": "1d", "range": "1d"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        data = res.json()
        price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return round(float(price), 2)
    except Exception as e:
        logger.error(f"VIX 조회 실패: {e}")
        return None


def fetch_usd_krw() -> float | None:
    """Yahoo Finance에서 USD/KRW 환율 조회."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/KRW=X"
    params = {"interval": "1d", "range": "1d"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        data = res.json()
        price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return round(float(price), 2)
    except Exception as e:
        logger.error(f"USD/KRW 조회 실패: {e}")
        return None


def fetch_fear_greed_index() -> dict | None:
    """CNN Fear & Greed Index 대용 — VIX 기반 간이 산출."""
    vix = fetch_vix()
    if vix is None:
        return None
    thresholds = VIX_THRESHOLDS
    if vix <= thresholds["extreme_greed"]:
        sentiment = "Extreme Greed"
    elif vix <= thresholds["greed"]:
        sentiment = "Greed"
    elif vix <= thresholds["neutral"]:
        sentiment = "Neutral"
    elif vix <= thresholds["fear"]:
        sentiment = "Fear"
    else:
        sentiment = "Extreme Fear"
    return {"vix": vix, "sentiment": sentiment}


def run() -> dict:
    """
    스케줄러에서 호출하는 진입점.
    Returns: {"vix": float, "usd_krw": float, "sentiment": str}
    """
    logger.info("=== 매크로 지표 수집기 실행 ===")

    vix = fetch_vix()
    usd_krw = fetch_usd_krw()
    fg = fetch_fear_greed_index()

    result = {
        "vix": vix,
        "usd_krw": usd_krw,
        "sentiment": fg["sentiment"] if fg else None,
    }

    logger.info(f"  VIX: {vix}")
    logger.info(f"  USD/KRW: {usd_krw}")
    logger.info(f"  Sentiment: {result['sentiment']}")

    return result
