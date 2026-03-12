"""
기능:
  1. 하드코딩된 스케줄에 따라 크롤러·전략을 자동 실행
  2. AI(Gemini/GPT)로 아침 시장 브리핑 & 종가 예측, 장 마감 후 리뷰
  3. 토큰 자동 갱신, 에러 복구
  4. 당일 마지막 스케줄 작업 완료 후 자동 종료
"""

import sys
import json
import time
import logging
import datetime
import traceback
import threading
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

#  AI 클라이언트 초기화 (Gemini / GPT) — 브리핑 & 예측용
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
    logger.warning("AI API 사용 불가 — 브리핑/예측 기능이 비활성화됩니다")


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
    "ai_briefing": {
        "module": "scrapers.ai_market_briefing",
        "description": "AI 시장 브리핑 & 종가 예측",
        "type": "ai",
    },
    "ai_closing_review": {
        "module": "scrapers.ai_market_briefing",
        "description": "AI 장 마감 리뷰 (예측 vs 실제 비교)",
        "type": "ai",
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
    "stock_scrapping": {
        "module": "scrapers.stock_scrapping",
        "description": "장중 주가 OHLCV 수집",
        "type": "scraper",
        "run_in_background": True,
    },
    # __INTEGRATE_MARKER_TASK_REGISTRY__
}

# 기본 스케줄 (settings.py 하드코딩)
DEFAULT_SCHEDULE = cfg.DEFAULT_SCHEDULE



#  작업 실행 엔진

_macro_cache: dict = {}
_background_tasks: dict[str, threading.Thread] = {}
_prefetched_kis_token: str | None = None


def _start_background_task(task_id: str, mod, *, prefetched_token: str | None = None) -> dict:
    existing = _background_tasks.get(task_id)
    if existing and existing.is_alive():
        logger.info(f"⏭ 백그라운드 작업 이미 실행 중: {task_id}")
        return {"status": "already_running", "background": True}

    def _runner():
        try:
            mod.run(prefetched_token=prefetched_token)
            logger.info(f"✔ 백그라운드 작업 완료: {task_id}")
        except Exception as exc:
            logger.error(f"✘ 백그라운드 작업 실패: {task_id} — {exc}")
            logger.debug(traceback.format_exc())
        finally:
            _background_tasks.pop(task_id, None)

    thread = threading.Thread(target=_runner, name=f"task-{task_id}", daemon=True)
    _background_tasks[task_id] = thread
    thread.start()
    logger.info(f"↪ 백그라운드 작업 시작: {task_id}")
    return {"status": "started", "background": True}


def execute_task(task_id: str, dry_run: bool | None = None) -> dict | None:
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

        if task_info.get("run_in_background"):
            return _start_background_task(task_id, mod, prefetched_token=_prefetched_kis_token)

        # AI 브리핑/리뷰 작업
        if task_id == "ai_briefing":
            ai_func = _ai_generate if (AI_AVAILABLE and cfg.USE_AI_BRIEFING) else None
            result = mod.run(macro_data=_macro_cache, ai_generate=ai_func)
        elif task_id == "ai_closing_review":
            ai_func = _ai_generate if (AI_AVAILABLE and cfg.USE_AI_BRIEFING) else None
            result = mod.run_closing_review(ai_generate=ai_func)
        # strategy 모듈은 macro_data와 dry_run 파라미터 전달
        elif task_info["type"] == "strategy":
            result = mod.run(macro_data=_macro_cache, dry_run=dry_run)
        else:
            result = mod.run()

        # 매크로 지표 결과 캐시
        if task_id == "macro_indicators" and result:
            _macro_cache = result

        logger.info(f"✔ 작업 완료: {task_id}")
        return result

    except Exception as e:
        logger.error(f"✘ 작업 실패: {task_id} — {e}")
        logger.debug(traceback.format_exc())
        return None

#  스케줄 등록 및 메인 루프
def _parse_schedule_time(time_str: str | None) -> datetime.time | None:
    if not time_str:
        return None
    try:
        hour, minute = map(int, time_str.split(":"))
        return datetime.time(hour, minute)
    except (TypeError, ValueError):
        logger.warning(f"잘못된 스케줄 시간 형식, 즉시 실행 판단 건너뜀: {time_str}")
        return None


def _should_start_immediately(item: dict, now: datetime.datetime | None = None) -> bool:
    task_id = item.get("task")
    if not task_id or not cfg.ENABLE_TASKS.get(task_id, False):
        return False

    task_info = TASK_REGISTRY.get(task_id, {})
    if not task_info.get("run_in_background"):
        return False

    module_path = task_info.get("module")
    if module_path:
        try:
            mod = import_module(module_path)
            should_start_now = getattr(mod, "should_start_now", None)
            if callable(should_start_now) and not should_start_now(now=now):
                return False
        except Exception as exc:
            logger.warning(f"즉시 실행 조건 확인 실패, 스케줄 기준만 사용: {task_id} ({exc})")

    start_time = _parse_schedule_time(item.get("time"))
    end_time = _parse_schedule_time(item.get("force_kill_time"))
    if start_time is None or end_time is None:
        return False

    now = now or datetime.datetime.now()
    current_time = now.time().replace(second=0, microsecond=0)
    return start_time <= current_time <= end_time


def trigger_startup_tasks(schedule_items: list[dict], now: datetime.datetime | None = None):
    now = now or datetime.datetime.now()
    for item in schedule_items:
        if not _should_start_immediately(item, now):
            continue

        task_id = item["task"]
        logger.info(
            f"⏩ 현재 시각 {now.strftime('%H:%M')} 이(가) {task_id} 활성 구간 안에 있어 즉시 시작합니다 "
            f"({item.get('time')}~{item.get('force_kill_time')})"
        )
        execute_task(task_id)


def _get_last_task_time(schedule_items: list[dict]) -> str | None:
    """활성화된 작업 중 마지막 예정 시간을 반환."""
    active_times = [
        item.get("force_kill_time") or item["time"] for item in schedule_items
        if cfg.ENABLE_TASKS.get(item["task"], False) and (item.get("force_kill_time") or item.get("time"))
    ]
    return max(active_times) if active_times else None


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
        force_kill_time = item.get("force_kill_time")
        if force_kill_time:
            logger.info(f"  📅 {t} → {task_id} ({desc}, 종료 {force_kill_time})")
        else:
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
    print("║  마지막 작업 완료 후 자동 종료됩니다.             ║")
    print("║  수동 종료: Ctrl+C                               ║")
    print("╚══════════════════════════════════════════════════╝")

    mode = "🔴 실전 주문" if not cfg.DRY_RUN else "🟢 시뮬레이션 (DRY RUN)"
    active = [t for t, on in cfg.ENABLE_TASKS.items() if on]
    print(f"\n  모드: {mode}")
    print(f"  활성 작업: {', '.join(active)}")
    print(f"  AI 브리핑/예측: {'ON' if cfg.USE_AI_BRIEFING else 'OFF'}")
    print()


def main():
    global _prefetched_kis_token

    print_banner()

    # 1. KIS API 토큰 사전 발급
    try:
        from utils.kis_api import get_access_token
        _prefetched_kis_token = get_access_token()
        logger.info("KIS 토큰 사전 발급 완료")
    except Exception as e:
        logger.warning(f"KIS 토큰 사전 발급 실패 (API 키 미설정?): {e}")

    # 2. 하드코딩 스케줄 적용
    logger.info("=" * 50)
    logger.info("오늘의 스케줄 등록 중 (설정 기반)...")
    today_schedule = DEFAULT_SCHEDULE

    # 3. 스케줄 등록
    if today_schedule:
        logger.info("─" * 50)
        logger.info("등록된 스케줄:")
        register_schedule(today_schedule)
        trigger_startup_tasks(today_schedule)
        logger.info("─" * 50)
    else:
        logger.info("실행할 작업이 없습니다. 종료합니다.")
        return

    # 4. 마지막 작업 시간 계산 (자동 종료용)
    last_time_str = _get_last_task_time(today_schedule)
    if last_time_str:
        # 마지막 작업 시간 + 5분 여유 후 종료
        h, m = map(int, last_time_str.split(":"))
        shutdown_dt = datetime.datetime.combine(
            datetime.date.today(),
            datetime.time(h, m),
        ) + datetime.timedelta(minutes=5)
        logger.info(f"자동 종료 예정: {shutdown_dt.strftime('%H:%M')} (마지막 작업 {last_time_str} + 5분)")
    else:
        logger.info("활성화된 작업이 없습니다. 종료합니다.")
        return

    # 5. 메인 루프 (마지막 작업 후 자동 종료)
    logger.info("메인 루프 시작 — 스케줄된 작업을 대기합니다...")

    while True:
        try:
            schedule.run_pending()

            # 마지막 작업 시간 + 5분 경과 → 자동 종료 ->서버 자동 종료 기능
            now = datetime.datetime.now()
            if now >= shutdown_dt:
                logger.info("=" * 50)
                logger.info("오늘의 모든 스케줄 작업이 완료되었습니다. 자동 종료합니다.")
                break

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
