"""
기능:
  1. Gemini API로 오늘의 스케줄을 생성/확인
  2. 스케줄에 따라 크롤러·전략을 자동 실행
  3. 각 작업 결과를 Gemini에게 전달 → 다음 액션 판단
  4. 토큰 자동 갱신, 에러 복구, 24시간 무중단 운영
"""

import sys
import json
import time
import logging
import datetime
import traceback
from importlib import import_module

import schedule
import settings as cfg

# logging 설정
LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s — %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scheduler.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("scheduler")

# ═════════════════════════════════════════════════
#  AI 클라이언트 초기화 (Gemini / GPT)
# ═════════════════════════════════════════════════
_gemini_model = None
_openai_client = None

if cfg.AI_PROVIDER == "gemini":
    try:
        import google.generativeai as genai
        genai.configure(api_key=cfg.GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(cfg.GEMINI_MODEL)
        logger.info(f"Gemini API 연결 완료 (모델: {cfg.GEMINI_MODEL})")
    except Exception as e:
        logger.warning(f"Gemini API 초기화 실패: {e}")
elif cfg.AI_PROVIDER == "gpt":
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=cfg.GPT_API_KEY)
        logger.info(f"OpenAI API 연결 완료 (모델: {cfg.GPT_MODEL})")
    except Exception as e:
        logger.warning(f"OpenAI API 초기화 실패: {e}")

AI_AVAILABLE = _gemini_model is not None or _openai_client is not None
if not AI_AVAILABLE:
    logger.warning("AI API 사용 불가 — 기본 스케줄로 동작합니다")


def _ai_generate(prompt: str) -> str | None:
    """설정된 AI_PROVIDER에 따라 Gemini 또는 GPT를 호출하여 응답 텍스트를 반환."""
    try:
        if cfg.AI_PROVIDER == "gemini" and _gemini_model:
            response = _gemini_model.generate_content(prompt)
            return response.text.strip()
        elif cfg.AI_PROVIDER == "gpt" and _openai_client:
            response = _openai_client.chat.completions.create(
                model=cfg.GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"AI 호출 실패 ({cfg.AI_PROVIDER}): {e}")
    return None


#  모듈 레지스트리: 실행 가능한 작업 목록

TASK_REGISTRY = {
    "krx_disclosure": {
        "module": "scrapers.krx_kind_crawler",
        "description": "KRX KIND 공시 크롤링",
        "type": "scraper",
    },
    "macro_indicators": {
        "module": "scrapers.macro_indicator_scraper",
        "description": "매크로 지표 수집 (VIX, 환율, 심리지수)",
        "type": "scraper",
    },
    "factor_momentum": {
        "module": "strategies.factor_momentum_allocation",
        "description": "팩터/모멘텀 기반 비중 조절 전략",
        "type": "strategy",
    },
    "hrl_allocation": {
        "module": "strategies.hrl_asset_allocation",
        "description": "HRL 기반 자산 배분 전략",
        "type": "strategy",
    },
}

#  기본 스케줄 (settings.py에서 관리)
DEFAULT_SCHEDULE = cfg.DEFAULT_SCHEDULE


#  Gemini 추천 스케줄 생성
def ask_gemini_for_schedule() -> list[dict] | None:
    if not AI_AVAILABLE or not cfg.USE_AI_SCHEDULER:
        return None

    today = datetime.date.today()
    weekday = today.strftime("%A")

    # 활성화된 작업만 Gemini에게 전달
    active_tasks = {
        tid: info for tid, info in TASK_REGISTRY.items()
        if cfg.ENABLE_TASKS.get(tid, False)
    }
    task_list_str = "\n".join(
        f"  - {tid}: {info['description']} (type: {info['type']})"
        for tid, info in active_tasks.items()
    )

    prompt = f"""당신은 퀀트 트레이딩 시스템의 스케줄 관리자입니다.
오늘은 {today.isoformat()} ({weekday}) 입니다.
한국 주식시장 운영 시간: 09:00~15:30 (주말/공휴일 휴무)

실행 가능한 작업 목록:
{task_list_str}

아래 규칙을 따라 오늘의 실행 스케줄을 JSON 배열로 만들어 주세요:
1. 주말(토/일)이면 빈 배열 [] 반환
2. scraper 타입은 장 시작 전(08:50), 점심(12:00), 장 마감 후(15:35)에 실행
3. strategy 타입은 장 시작 직후(09:10~09:20)에 실행
4. 각 항목: {{"time": "HH:MM", "task": "task_id", "description": "설명"}}

JSON 배열만 반환하세요. 다른 텍스트 없이 순수 JSON만 출력하세요.
"""

    try:
        text = _ai_generate(prompt)
        if text is None:
            return None
        # Markdown 코드 블록 제거
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        result = json.loads(text)
        if isinstance(result, list):
            logger.info(f"AI 스케줄 수신: {len(result)}개 작업")
            return result
    except Exception as e:
        logger.error(f"AI 스케줄 생성 실패: {e}")
    return None


def ask_gemini_for_action(task_name: str, task_result: dict) -> str | None:
    """
    작업 결과를 AI에게 보내고, 후속 조치 의견을 받음.
    """
    if not AI_AVAILABLE or not cfg.USE_AI_ADVISOR:
        return None

    result_summary = json.dumps(task_result, ensure_ascii=False, default=str)[:2000]

    prompt = f"""당신은 퀀트 트레이딩 시스템의 AI 어드바이저입니다.
방금 '{task_name}' 작업이 완료되었습니다.

결과:
{result_summary}

위 결과를 분석하고, 다음 중 하나로 답변하세요:
1. 특별한 조치가 필요 없으면: "ACTION: NONE"
2. 추가 작업이 필요하면: "ACTION: RUN <task_id>" (task_id는 등록된 작업 ID)
3. 주의가 필요하면: "ACTION: ALERT <간단한 이유>"

한 줄로 답변하세요.
"""

    try:
        text = _ai_generate(prompt)
        if text:
            logger.info(f"AI 어드바이저 응답: {text}")
            return text
    except Exception as e:
        logger.error(f"AI 어드바이저 호출 실패: {e}")
    return None


# ═══════════════════════════════════════════════════
#  작업 실행 엔진
# ═══════════════════════════════════════════════════
# 최근 매크로 데이터 캐시 (전략에서 참조)
_macro_cache: dict = {}


def execute_task(task_id: str, dry_run: bool | None = None) -> dict | None:
    """
    레지스트리에 등록된 작업을 동적으로 로드하여 실행.
    dry_run이 None이면 settings.DRY_RUN 값을 사용.
    """
    global _macro_cache

    if dry_run is None:
        dry_run = cfg.DRY_RUN

    if task_id not in TASK_REGISTRY:
        logger.error(f"알 수 없는 작업 ID: {task_id}")
        return None

    if not cfg.ENABLE_TASKS.get(task_id, False):
        logger.info(f"⏭ 작업 비활성화 상태, 건너뜀: {task_id}")
        return None

    task_info = TASK_REGISTRY[task_id]
    module_path = task_info["module"]

    try:
        logger.info(f"▶ 작업 실행: {task_id} ({task_info['description']})")
        mod = import_module(module_path)

        # strategy 모듈은 macro_data와 dry_run 파라미터 전달
        if task_info["type"] == "strategy":
            result = mod.run(macro_data=_macro_cache, dry_run=dry_run)
        else:
            result = mod.run()

        # 매크로 지표 결과 캐시
        if task_id == "macro_indicators" and result:
            _macro_cache = result

        logger.info(f"✔ 작업 완료: {task_id}")

        # Gemini에게 결과 전달 → 후속 조치 판단
        action = ask_gemini_for_action(task_id, result)
        if action and "ACTION: RUN" in action:
            parts = action.split("ACTION: RUN")
            if len(parts) > 1:
                next_task = parts[1].strip()
                if next_task in TASK_REGISTRY:
                    logger.info(f"AI 권고에 따라 추가 실행: {next_task}")
                    execute_task(next_task, dry_run=dry_run)
        elif action and "ACTION: ALERT" in action:
            logger.warning(f"⚠ AI 알림: {action}")

        return result

    except Exception as e:
        logger.error(f"✘ 작업 실패: {task_id} — {e}")
        logger.debug(traceback.format_exc())
        return None


# ═══════════════════════════════════════════════════
#  스케줄 등록 및 메인 루프
# ═══════════════════════════════════════════════════
def register_schedule(schedule_items: list[dict]):
    """schedule 라이브러리에 작업 등록. 비활성화 작업은 자동 제외."""
    schedule.clear()
    for item in schedule_items:
        t = item["time"]
        task_id = item["task"]
        desc = item.get("description", "")
        if not cfg.ENABLE_TASKS.get(task_id, False):
            logger.info(f"  ⏭ {t} → {task_id} (비활성화, 건너뜀)")
            continue
        schedule.every().day.at(t).do(execute_task, task_id=task_id)
        logger.info(f"  📅 {t} → {task_id} ({desc})")


def print_banner():
    """시작 배너 출력."""
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║      QUANT TRADING SYSTEM — CONTROL TOWER       ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  main_scheduler.py 를 실행하면 모든 작업이       ║")
    print("║  자동으로 스케줄에 따라 동작합니다.               ║")
    print("║                                                  ║")
    print("║  종료: Ctrl+C                                    ║")
    print("║  tmux에서 분리: Ctrl+B → D                       ║")
    print("╚══════════════════════════════════════════════════╝")

    mode = "🔴 실전 주문" if not cfg.DRY_RUN else "🟢 시뮬레이션 (DRY RUN)"
    active = [t for t, on in cfg.ENABLE_TASKS.items() if on]
    print(f"\n  모드: {mode}")
    print(f"  활성 작업: {', '.join(active)}")
    print(f"  Gemini 스케줄러: {'ON' if cfg.USE_AI_SCHEDULER else 'OFF'}")
    print(f"  Gemini 어드바이저: {'ON' if cfg.USE_AI_ADVISOR else 'OFF'}")
    print()


def main():
    print_banner()

    # 1. KIS API 토큰 사전 발급
    try:
        from utils.kis_api import get_access_token
        token = get_access_token()
        logger.info("KIS 토큰 사전 발급 완료")
    except Exception as e:
        logger.warning(f"KIS 토큰 사전 발급 실패 (API 키 미설정?): {e}")

    # 2. Gemini에게 오늘 스케줄 요청
    logger.info("=" * 50)
    logger.info("오늘의 스케줄 생성 중...")
    gemini_schedule = ask_gemini_for_schedule()

    if gemini_schedule is not None:
        # AI 스케줄 유효성 검증
        valid_schedule = [
            item for item in gemini_schedule
            if item.get("task") in TASK_REGISTRY and item.get("time")
        ]
        if valid_schedule:
            logger.info(f"AI 스케줄 적용 ({len(valid_schedule)}개 작업)")
            today_schedule = valid_schedule
        else:
            logger.warning("AI 스케줄이 비어있음 (주말/공휴일?) — 대기 모드")
            today_schedule = []
    else:
        logger.info("기본 스케줄 적용")
        today_schedule = DEFAULT_SCHEDULE

    # 3. 스케줄 등록
    if today_schedule:
        logger.info("─" * 50)
        logger.info("등록된 스케줄:")
        register_schedule(today_schedule)
        logger.info("─" * 50)
    else:
        logger.info("오늘 실행할 작업이 없습니다. 자정까지 대기합니다.")

    # 4. 메인 루프 (24시간 무중단)
    logger.info("메인 루프 시작 — 스케줄된 작업을 대기합니다...")

    last_reschedule_date = datetime.date.today()

    while True:
        try:
            schedule.run_pending()

            # 날짜가 바뀌면 스케줄 재생성
            today = datetime.date.today()
            if today != last_reschedule_date:
                logger.info("=" * 50)
                logger.info(f"새로운 날짜 감지: {today.isoformat()}")
                logger.info("스케줄 재생성 중...")

                gemini_schedule = ask_gemini_for_schedule()
                if gemini_schedule is not None:
                    valid_schedule = [
                        item for item in gemini_schedule
                        if item.get("task") in TASK_REGISTRY and item.get("time")
                    ]
                    if valid_schedule:
                        register_schedule(valid_schedule)
                    else:
                        schedule.clear()
                        logger.info("오늘 실행할 작업 없음 (주말/공휴일)")
                else:
                    register_schedule(DEFAULT_SCHEDULE)

                last_reschedule_date = today

            time.sleep(cfg.SCHEDULE_CHECK_INTERVAL_SEC)

        except KeyboardInterrupt:
            logger.info("사용자에 의해 종료됨 (Ctrl+C)")
            break
        except Exception as e:
            logger.error(f"메인 루프 에러: {e}")
            logger.debug(traceback.format_exc())
            time.sleep(cfg.ERROR_RETRY_DELAY_SEC)


if __name__ == "__main__":
    main()
