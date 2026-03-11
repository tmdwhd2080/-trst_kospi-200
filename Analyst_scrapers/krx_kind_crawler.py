"""
KRX KIND 공시 모니터링 크롤러
- 최신 공시 목록 수집
- 키워드 기반 필터링 (유상증자, 무상증자, 합병, 분할 등)
"""

# ═══════════════════════════════════════════════════
#  ⚙ 크롤러 개별 설정 — 이 영역의 값만 수정하세요
# ═══════════════════════════════════════════════════
CRAWL_PAGE_SIZE = 100           # 한 번에 가져올 공시 건수
ALERT_KEYWORDS = [
    "유상증자", "무상증자", "합병", "분할", "자사주",
    "배당", "대규모내부거래", "공개매수", "상장폐지",
    "액면분할", "전환사채", "신주인수권",
]
# ═══════════════════════════════════════════════════

import logging
import datetime
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

KIND_URL = "https://kind.krx.co.kr/disclosure/todaydisclosure.do"


def fetch_today_disclosures() -> list[dict]:
    """
    오늘 KRX KIND 공시 목록을 크롤링하여 반환.
    [{
        "time": "09:01",
        "company": "삼성전자",
        "title": "주요사항보고서(자기주식취득결정)",
        "url": "...",
    }, ...]
    """
    today_str = datetime.date.today().strftime("%Y%m%d")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    params = {
        "method": "searchTodayDisclosureSub",
        "currentPageSize": str(CRAWL_PAGE_SIZE),
        "pageIndex": "1",
        "orderMode": "0",
        "orderStat": "D",
        "forward": "todaydisclosure_sub",
        "chose": "S",
        "todayFlag": "Y",
        "selDate": today_str,
    }

    results = []
    try:
        res = requests.get(KIND_URL, headers=headers, params=params, timeout=15)
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, "lxml")

        rows = soup.select("table tbody tr")
        for row in rows:
            cols = row.select("td")
            if len(cols) < 4:
                continue
            disc_time = cols[0].get_text(strip=True)
            company = cols[1].get_text(strip=True)
            title_tag = cols[2].select_one("a")
            title = title_tag.get_text(strip=True) if title_tag else cols[2].get_text(strip=True)
            link = title_tag["href"] if title_tag and title_tag.has_attr("href") else ""

            results.append({
                "time": disc_time,
                "company": company,
                "title": title,
                "url": f"https://kind.krx.co.kr{link}" if link.startswith("/") else link,
            })

        logger.info(f"KRX 공시 {len(results)}건 수집 완료 ({today_str})")
    except Exception as e:
        logger.error(f"KRX 공시 크롤링 실패: {e}")

    return results


def filter_important_disclosures(disclosures: list[dict]) -> list[dict]:
    """키워드 기반으로 중요 공시만 필터링."""
    important = []
    for d in disclosures:
        for kw in ALERT_KEYWORDS:
            if kw in d["title"]:
                important.append({**d, "matched_keyword": kw})
                break
    return important


def run() -> dict:
    """
    스케줄러에서 호출하는 진입점.
    Returns: {"total": int, "important": list[dict], "all": list[dict]}
    """
    logger.info("=== KRX 공시 크롤러 실행 ===")
    all_disc = fetch_today_disclosures()
    important = filter_important_disclosures(all_disc)

    if important:
        logger.info(f"  ⚠ 중요 공시 {len(important)}건 감지:")
        for d in important:
            logger.info(f"    [{d['matched_keyword']}] {d['company']} - {d['title']}")
    else:
        logger.info("  중요 공시 없음")

    return {"total": len(all_disc), "important": important, "all": all_disc}
