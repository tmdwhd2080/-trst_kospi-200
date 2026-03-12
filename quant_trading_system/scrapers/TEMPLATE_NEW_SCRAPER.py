"""
[새 스크래퍼] 스크래퍼 이름을 여기에 작성
──────────────────────────────────────────
스크래퍼 설명을 여기에 작성.

⚠ 이 파일을 복사하여 새 스크래퍼를 작성하세요.
  1. 이 파일을 scrapers/ 디렉토리에 복사 & 이름 변경
     예: scrapers/naver_news_scraper.py
  2. 아래 TASK_META를 수정 (task_id, description, schedule 등)
  3. run() 함수에 크롤링/수집 로직 구현
  4. 통합 스크립트 실행:
     python scrapers/integrate_scraper.py scrapers/<파일명>.py

★ TASK_META의 schedule에 time(시작 시간)과 force_kill_time(강제 종료 시간)을
  반드시 작성해야 합니다.
"""

#  작업 메타데이터 — integrate_scraper.py가 읽어갑니다
#  이거 있어야지 integrate_scraper.py가 알아서 main_scheduler.py와 settings.py에 등록해주기 때문에 작성 필수임.
TASK_META = {
    "task_id": "my_scraper_name",            # 고유 작업 ID (영문 소문자 + 언더스코어)
    "description": "나의 새 스크래퍼 설명",    # 작업 설명 (한글 가능)
    "type": "scraper",                        # ← "scraper" 고정 (변경 금지)
    "enabled": True,                          # 활성화 여부 (True/False)
    "schedule": [
        {
            "time": "09:00",                  #  실행 시작 시간 (HH:MM, 필수)
            "force_kill_time": "09:30",       #  강제 종료 시간 (HH:MM, 필수)
            "description": "나의 스크래퍼 실행",
        },
        # 하루에 해당 전략을 여러 번 실행하려면 항목을 추가:
        # {
        #     "time": "14:00",
        #     "force_kill_time": "14:30",
        #     "description": "오후 나의 스크래퍼 실행",
        # },
    ],
}

# ═══════════════════════════════════════════════════
#  ⚙ 스크래퍼 개별 설정 — 이 영역의 값만 수정하세요
# ═══════════════════════════════════════════════════
# 예: TARGET_URL = "https://example.com/api/data"
# 예: KEYWORDS = ["키워드1", "키워드2"]
# 예: PAGE_SIZE = 100
# ═══════════════════════════════════════════════════


# 이거 주고 llm 한테 맞춰 달라고 하면 맞춰줌 -> 되도록 copilot 쓰면 그냥 딸깍이긴함,...
import logging

logger = logging.getLogger(__name__)


def run() -> dict:
    """
    스케줄러에서 호출하는 진입점.

    ※ 반드시 이 함수 시그니처를 유지하세요:
      - 함수명: run
      - 파라미터: 없음
      - 반환: dict

    Returns:
        dict: 수집 결과 (자유 형식)
    """
    logger.info("=== 나의 스크래퍼 실행 ===")

    # TODO: 크롤링/수집 로직을 여기에 구현하세요
    # 예:
    #   import requests
    #   res = requests.get(TARGET_URL, timeout=10)
    #   data = res.json()
    #   processed = [item for item in data if any(kw in item["title"] for kw in KEYWORDS)]
    #   logger.info(f"  수집 완료: {len(processed)}건")
    #   return {"count": len(processed), "data": processed}

    result = {}
    logger.info(f"  수집 완료: {result}")
    return result
