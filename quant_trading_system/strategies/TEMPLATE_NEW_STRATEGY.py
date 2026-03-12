"""
[새 전략] 전략 이름을 여기에 작성
──────────────────────────────────────────
전략 설명을 여기에 작성.

⚠ 이 파일을 복사하여 새 전략을 작성하세요.
  1. 이 파일을 strategies/ 디렉토리에 복사 & 이름 변경
     예: strategies/mean_reversion_strategy.py
  2. 아래 TASK_META를 수정 (task_id, description, schedule 등)
  3. run() 함수에 전략 로직 구현
  4. 통합 스크립트 실행:
     python strategies/integrate_strategy.py strategies/<파일명>.py

★ TASK_META의 schedule에 time(시작 시간)과 force_kill_time(강제 종료 시간)을
  반드시 작성해야 합니다.
"""

#  작업 메타데이터 — integrate_strategy.py가 읽어갑니다
#     반드시 모든 필드를 작성하세요! -> 얘도 이거 Tjdiwl integrate_strategy.py가 알아서 main_scheduler.py와 settings.py에 등록해주기 때문에 작성 필수임.
TASK_META = {
    "task_id": "my_strategy_name",           # 고유 작업 ID (영문 소문자 + 언더스코어)
    "description": "나의 새 전략 설명",        # 작업 설명 (한글 가능)
    "type": "strategy",                       # ← "strategy" 고정 (변경 금지)
    "enabled": True,                          # 활성화 여부 (True/False)
    "schedule": [
        {
            "time": "09:30",                  # ★ 실행 시작 시간 (HH:MM, 필수)
            "force_kill_time": "10:00",       # ★ 강제 종료 시간 (HH:MM, 필수)
            "description": "나의 전략 실행",
        },
    ],
}

# ═══════════════════════════════════════════════════
#  ⚙ 전략 개별 설정 — 이 영역의 값만 수정하세요
# ═══════════════════════════════════════════════════
# 예: UNIVERSE = ["005930", "000660", "035420"]
# 예: MAX_STOCKS = 5
# 예: ORDER_QTY = 1
# 예: ALLOCATION_PCT = 0.30
# ═══════════════════════════════════════════════════


# 얘도 이거 llm한테 주거나 아니면 맘대로 쓰고 copilot 딸깍 해도 되긴 함
import logging
from utils.kis_api import get_current_price, buy_limit_order

logger = logging.getLogger(__name__)


def run(macro_data: dict | None = None, dry_run: bool = True) -> dict:

    logger.info("=== 나의 전략 실행 ===")

    # TODO: 전략 로직을 여기에 구현하세요
    # 예:
    #   price = get_current_price("005930")
    #   if dry_run:
    #       logger.info(f"[DRY RUN] 매수: 005930 / 1주 / {price}원")
    #   else:
    #       buy_limit_order("005930", 1, price)

    result = {"orders": []}
    logger.info(f"  전략 실행 완료: {result}")
    return result
