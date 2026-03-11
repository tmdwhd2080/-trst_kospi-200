"""
AI 시장 브리핑 & 예측 시스템
──────────────────────────────
- 아침: 매크로 데이터를 기반으로 AI가 시장 브리핑 + 종가 예측
- 장 마감 후: 실제 종가 vs 예측 비교 → 학습 로그 저장
- 로그 파일: logs/ai_predictions.jsonl (매일 누적)
"""

import os
import json
import logging
import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# 로그 디렉토리
LOG_DIR = Path(__file__).parent.parent / "logs"
PREDICTION_LOG = LOG_DIR / "ai_predictions.jsonl"

# 오늘의 예측 캐시 (메모리)
_today_prediction: dict = {}


def _load_recent_reviews(n: int = 5) -> str:
    """최근 n일간의 예측 리뷰 로그를 읽어 문자열로 반환 (학습 컨텍스트)."""
    if not PREDICTION_LOG.exists():
        return "과거 예측 기록 없음 (첫 실행)"

    lines = []
    try:
        with open(PREDICTION_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    except Exception:
        return "과거 예측 기록 읽기 실패"

    # 마지막 n개만
    recent = lines[-n:]
    summaries = []
    for raw in recent:
        try:
            entry = json.loads(raw)
            if entry.get("type") == "closing_review":
                date = entry.get("date", "?")
                pred = entry.get("predictions", {})
                actual = entry.get("actuals", {})
                errors = entry.get("errors", {})
                summaries.append(
                    f"  {date}: 예측 KOSPI={pred.get('kospi')}, 실제={actual.get('kospi')}, "
                    f"오차={errors.get('kospi')} / "
                    f"KOSDAQ 예측={pred.get('kosdaq')}, 실제={actual.get('kosdaq')}, "
                    f"오차={errors.get('kosdaq')} / "
                    f"KOSPI200 예측={pred.get('kospi200')}, 실제={actual.get('kospi200')}, "
                    f"오차={errors.get('kospi200')}"
                )
        except Exception:
            continue

    if not summaries:
        return "과거 예측 기록 없음"
    return "\n".join(summaries)


def run(macro_data: dict | None = None, ai_generate=None) -> dict:
    """
    아침 브리핑: 매크로 데이터 기반 AI 시장 분석 + 종가 예측.

    Args:
        macro_data: macro_indicator_scraper 수집 결과
        ai_generate: main_scheduler의 _ai_generate 함수 참조
    Returns:
        {"briefing": str, "predictions": {"kospi": float, "kosdaq": float, "kospi200": float}}
    """
    global _today_prediction
    logger.info("=== AI 시장 브리핑 실행 ===")

    if ai_generate is None:
        logger.warning("AI 함수가 전달되지 않음 — 브리핑 건너뜀")
        return {"briefing": None, "predictions": {}}

    # 매크로 데이터 요약
    if macro_data:
        macro_summary = json.dumps(macro_data, ensure_ascii=False, default=str)
    else:
        macro_summary = "매크로 데이터 없음 (수집 전이거나 실패)"

    # 과거 예측 기록 (학습 컨텍스트)
    past_reviews = _load_recent_reviews(5)

    today = datetime.date.today().isoformat()
    prompt = f"""당신은 한국 주식시장 전문 애널리스트입니다.
오늘 날짜: {today}

[오늘의 매크로 데이터]
{macro_summary}

[최근 예측 vs 실제 기록 (학습 참고용)]
{past_reviews}

위 데이터를 분석하여 아래 두 가지를 수행하세요:

1. **시장 브리핑**: 오늘의 매크로 상황을 간결하게 요약하고, 시장에 미칠 영향을 분석하세요. (5줄 이내)

2. **종가 예측**: 아래 3개 지수의 오늘 종가를 소수점 2자리까지 예측하세요.
   과거 예측 기록을 참고하여 오차를 줄이려고 노력하세요.

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "briefing": "시장 브리핑 텍스트",
  "reasoning": "예측 근거 (어떤 요인을 고려했는지)",
  "predictions": {{
    "kospi": 2650.00,
    "kosdaq": 850.00,
    "kospi200": 355.00
  }}
}}
"""

    try:
        text = ai_generate(prompt)
        if text is None:
            logger.error("AI 응답 없음")
            return {"briefing": None, "predictions": {}}

        # Markdown 코드블록 제거
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]

        result = json.loads(text)

        briefing = result.get("briefing", "")
        reasoning = result.get("reasoning", "")
        predictions = result.get("predictions", {})

        logger.info(f"  📊 시장 브리핑: {briefing}")
        logger.info(f"  🧠 예측 근거: {reasoning}")
        logger.info(f"  📈 KOSPI 예측: {predictions.get('kospi')}")
        logger.info(f"  📈 KOSDAQ 예측: {predictions.get('kosdaq')}")
        logger.info(f"  📈 KOSPI200 예측: {predictions.get('kospi200')}")

        # 예측 캐시 저장 (장 마감 리뷰에서 사용)
        _today_prediction = {
            "date": today,
            "predictions": predictions,
            "reasoning": reasoning,
            "briefing": briefing,
        }

        # 로그 파일에 아침 예측 기록
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "type": "morning_prediction",
            "date": today,
            "timestamp": datetime.datetime.now().isoformat(),
            "macro_data": macro_data,
            "briefing": briefing,
            "reasoning": reasoning,
            "predictions": predictions,
        }
        with open(PREDICTION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, default=str) + "\n")

        return {"briefing": briefing, "predictions": predictions, "reasoning": reasoning}

    except json.JSONDecodeError as e:
        logger.error(f"AI 응답 JSON 파싱 실패: {e}")
        return {"briefing": None, "predictions": {}}
    except Exception as e:
        logger.error(f"AI 브리핑 실행 실패: {e}")
        return {"briefing": None, "predictions": {}}


def _fetch_index_closing_price(symbol: str) -> float | None:
    """Yahoo Finance에서 지수 종가 조회."""
    import requests
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": "1d", "range": "1d"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        data = res.json()
        price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return round(float(price), 2)
    except Exception as e:
        logger.error(f"지수 종가 조회 실패 ({symbol}): {e}")
        return None


def run_closing_review(ai_generate=None) -> dict:
    """
    장 마감 후 실행: 실제 종가 vs 예측 비교 → 로그 저장.

    Args:
        ai_generate: main_scheduler의 _ai_generate 함수 참조
    Returns:
        {"actuals": dict, "errors": dict, "review": str}
    """
    global _today_prediction
    logger.info("=== AI 장 마감 리뷰 실행 ===")

    today = datetime.date.today().isoformat()
    predictions = _today_prediction.get("predictions", {})

    if not predictions:
        logger.warning("오늘 아침 예측 데이터가 없습니다 — 리뷰 건너뜀")
        return {"actuals": {}, "errors": {}, "review": "예측 데이터 없음"}

    # 실제 종가 조회
    actuals = {
        "kospi": _fetch_index_closing_price("^KS11"),
        "kosdaq": _fetch_index_closing_price("^KQ11"),
        "kospi200": _fetch_index_closing_price("^KS200"),
    }

    # 오차 계산
    errors = {}
    for key in ["kospi", "kosdaq", "kospi200"]:
        pred = predictions.get(key)
        actual = actuals.get(key)
        if pred is not None and actual is not None:
            errors[key] = round(actual - pred, 2)
        else:
            errors[key] = None

    logger.info(f"  📊 KOSPI  — 예측: {predictions.get('kospi')}, 실제: {actuals['kospi']}, 오차: {errors['kospi']}")
    logger.info(f"  📊 KOSDAQ — 예측: {predictions.get('kosdaq')}, 실제: {actuals['kosdaq']}, 오차: {errors['kosdaq']}")
    logger.info(f"  📊 KOSPI200 — 예측: {predictions.get('kospi200')}, 실제: {actuals['kospi200']}, 오차: {errors['kospi200']}")

    # AI에게 자기 리뷰 요청 (학습용)
    review_text = ""
    if ai_generate:
        review_prompt = f"""당신은 한국 주식시장 전문 애널리스트입니다.
오늘({today}) 아침에 당신이 한 예측과 실제 결과를 비교합니다.

[아침 예측]
- 예측 근거: {_today_prediction.get('reasoning', 'N/A')}
- KOSPI 예측: {predictions.get('kospi')} → 실제: {actuals['kospi']} (오차: {errors['kospi']})
- KOSDAQ 예측: {predictions.get('kosdaq')} → 실제: {actuals['kosdaq']} (오차: {errors['kosdaq']})
- KOSPI200 예측: {predictions.get('kospi200')} → 실제: {actuals['kospi200']} (오차: {errors['kospi200']})

3줄 이내로 간결하게 자기 리뷰를 작성하세요:
1. 예측이 맞았다면 어떤 요소가 효과적이었는지
2. 틀렸다면 무엇을 놓쳤는지
3. 다음 예측에서 개선할 점
"""
        review_text = ai_generate(review_prompt) or ""
        if review_text:
            logger.info(f"  🧠 AI 자기 리뷰: {review_text}")

    # 로그 파일에 마감 리뷰 기록
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "type": "closing_review",
        "date": today,
        "timestamp": datetime.datetime.now().isoformat(),
        "predictions": predictions,
        "actuals": actuals,
        "errors": errors,
        "review": review_text,
    }
    with open(PREDICTION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False, default=str) + "\n")

    # 오늘 예측 캐시 초기화
    _today_prediction = {}

    return {"actuals": actuals, "errors": errors, "review": review_text}
