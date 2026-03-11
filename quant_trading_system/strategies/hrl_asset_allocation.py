"""
[예시 전략] 계층적 강화학습(HRL) 기반 자산 배분 전략
──────────────────────────────────────────────────────
- 상위 정책: 매크로 상황에 따라 공격/방어/중립 모드 결정
- 하위 정책: 각 모드에 맞는 종목별 비중 결정
- 실제 HRL 모델 학습/추론은 별도 구현 필요

⚠ 이 파일은 구조 예시입니다. 실제 모델은 직접 구현하세요.
"""

# ═══════════════════════════════════════════════════
#  ⚙ 전략 개별 설정 — 이 영역의 값만 수정하세요
# ═══════════════════════════════════════════════════
ASSET_UNIVERSE = {
    "equity":   ["005930", "000660", "035420"],     # 주식
    "bond_etf": ["148070"],                          # 채권 ETF
    "gold_etf": ["411060"],                          # 금 ETF
}
REGIME_RULES = {
    "Extreme Fear": {"equity": 0.2, "bond_etf": 0.5, "gold_etf": 0.3},
    "Fear":         {"equity": 0.3, "bond_etf": 0.4, "gold_etf": 0.3},
    "Neutral":      {"equity": 0.5, "bond_etf": 0.3, "gold_etf": 0.2},
    "Greed":        {"equity": 0.7, "bond_etf": 0.2, "gold_etf": 0.1},
    "Extreme Greed":{"equity": 0.8, "bond_etf": 0.1, "gold_etf": 0.1},
}
ORDER_QTY = 1                   # 종목당 기본 주문 수량
# ═══════════════════════════════════════════════════

import logging
from utils.kis_api import get_current_price, buy_limit_order, sell_limit_order

logger = logging.getLogger(__name__)


def determine_regime(macro_data: dict | None) -> str:
    """매크로 데이터 기반 현재 시장 국면 판단."""
    if macro_data is None:
        return "Neutral"
    return macro_data.get("sentiment", "Neutral")


def compute_target_allocation(regime: str) -> dict[str, float]:
    """국면에 따른 자산군별 목표 비중 반환."""
    return REGIME_RULES.get(regime, REGIME_RULES["Neutral"])


def run(macro_data: dict | None = None, dry_run: bool = True) -> dict:
    """
    전략 실행 진입점.
    Args:
        macro_data: 매크로 지표 수집기 결과
        dry_run: True면 주문 실행 안 함
    Returns:
        {"regime": str, "allocation": dict, "orders": list}
    """
    logger.info("=== HRL 자산 배분 전략 실행 ===")

    # 1. 시장 국면 판단
    regime = determine_regime(macro_data)
    logger.info(f"  현재 시장 국면: {regime}")

    # 2. 목표 비중 계산
    allocation = compute_target_allocation(regime)
    logger.info(f"  목표 비중: {allocation}")

    # 3. 각 자산군별 주문 생성
    orders = []
    for asset_class, weight in allocation.items():
        codes = ASSET_UNIVERSE.get(asset_class, [])
        per_stock_weight = weight / len(codes) if codes else 0

        for code in codes:
            price = get_current_price(code)
            if price is None:
                logger.warning(f"  {code} 가격 조회 실패, 건너뜀")
                continue

            qty = ORDER_QTY
            order_info = {
                "asset_class": asset_class,
                "stock_code": code,
                "price": price,
                "qty": qty,
                "weight": per_stock_weight,
            }

            if dry_run:
                logger.info(
                    f"  [DRY RUN] {asset_class} - {code}: "
                    f"{qty}주 @ {price}원 (비중 {per_stock_weight:.1%})"
                )
            else:
                result = buy_limit_order(code, qty, price)
                order_info["result"] = result

            orders.append(order_info)

    return {"regime": regime, "allocation": allocation, "orders": orders}
