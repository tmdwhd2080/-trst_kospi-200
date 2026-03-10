"""
[예시 전략] 팩터/모멘텀 기반 비중 조절 전략
─────────────────────────────────────────────
- KOSPI200 구성 종목 중 모멘텀 팩터 상위 N종목 선정
- 매크로 지표(VIX, 환율)에 따라 투자 비중 동적 조절
- 실제 주문은 KIS API를 통해 실행

⚠ 이 파일은 구조 예시입니다. 실제 전략 로직은 직접 구현하세요.
"""

import logging
from utils.kis_api import get_current_price, buy_limit_order, get_balance
import settings as cfg

logger = logging.getLogger(__name__)

UNIVERSE = cfg.FACTOR_UNIVERSE
MAX_STOCKS = cfg.FACTOR_MAX_STOCKS
ALLOCATION_PCT = cfg.FACTOR_ALLOCATION_PCT


def calculate_momentum_score(stock_code: str) -> float:
    """
    모멘텀 점수 산출 (예시: 현재가 기반 단순 스코어).
    실제로는 과거 N일 수익률, 거래량 등을 기반으로 계산.
    """
    price = get_current_price(stock_code)
    if price is None:
        return 0.0
    # 예시: 가격이 높을수록 높은 점수 (실전에서는 수익률 기반)
    return float(price)


def adjust_weight_by_macro(base_weight: float, macro_data: dict | None) -> float:
    """
    매크로 지표에 따라 투자 비중 조절.
    - VIX 30 이상 → 비중 절반으로 축소
    - VIX 15 이하 → 비중 20% 확대
    """
    if macro_data is None:
        return base_weight

    vix = macro_data.get("vix")
    if vix and vix >= 30:
        return base_weight * 0.5
    elif vix and vix <= 15:
        return min(base_weight * 1.2, 1.0)
    return base_weight


def run(macro_data: dict | None = None, dry_run: bool = True) -> dict:
    """
    전략 실행 진입점.
    Args:
        macro_data: 매크로 지표 수집기 결과
        dry_run: True면 주문 실행 안 함 (시뮬레이션만)
    Returns:
        {"selected": [...], "orders": [...]}
    """
    logger.info("=== 팩터/모멘텀 전략 실행 ===")

    # 1. 모멘텀 스코어 계산
    scores = {}
    for code in UNIVERSE:
        score = calculate_momentum_score(code)
        scores[code] = score
        logger.info(f"  {code} → 모멘텀 점수: {score}")

    # 2. 상위 N 종목 선정
    sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [code for code, _ in sorted_stocks[:MAX_STOCKS]]
    logger.info(f"  선정 종목: {selected}")

    # 3. 매크로 기반 비중 조절
    weight = adjust_weight_by_macro(ALLOCATION_PCT, macro_data)
    logger.info(f"  조정된 비중: {weight:.1%}")

    # 4. 주문 생성 (dry_run=True면 로그만)
    orders = []
    for code in selected:
        price = get_current_price(code)
        if price is None:
            continue
        qty = cfg.FACTOR_ORDER_QTY
        order_info = {"stock_code": code, "price": price, "qty": qty}

        if dry_run:
            logger.info(f"  [DRY RUN] 매수: {code} / {qty}주 / {price}원")
        else:
            result = buy_limit_order(code, qty, price)
            order_info["result"] = result
            logger.info(f"  [주문] 매수: {code} / {qty}주 / {price}원")

        orders.append(order_info)

    return {"selected": selected, "weight": weight, "orders": orders}
