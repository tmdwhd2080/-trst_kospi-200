"""
주문 기록 & 일별 수익률 로거
──────────────────────────────
전략의 run() 함수에서 호출하여 주문 내역과 일별 수익률을 CSV로 기록합니다.

사용법 (전략 내부):
    from performance.trade_logger import log_order, log_daily_return

    # 개별 주문 기록
    log_order("factor_momentum", "005930", "buy", 10, 55000)

    # 일별 수익률 기록 (장 마감 시점)
    log_daily_return("factor_momentum", portfolio_value=10_500_000, daily_pnl=50_000, daily_return_pct=0.48)
"""

import os
import csv
import datetime
import logging
import threading

logger = logging.getLogger(__name__)

# 저장 경로
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORDERS_CSV = os.path.join(_BASE_DIR, "data", "orders.csv")
DAILY_RETURNS_CSV = os.path.join(_BASE_DIR, "data", "daily_returns.csv")

_lock = threading.Lock()

ORDERS_HEADER = ["date", "time", "strategy_id", "stock_code", "action", "qty", "price", "amount"]
DAILY_HEADER = ["date", "strategy_id", "portfolio_value", "daily_pnl", "daily_return_pct"]


def _ensure_csv(filepath: str, header: list[str]):
    """CSV 파일이 없으면 헤더와 함께 생성."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def log_order(strategy_id: str, stock_code: str, action: str, qty: int, price: float):
    """
    개별 주문을 CSV에 기록.

    Args:
        strategy_id: 전략 ID (예: "factor_momentum")
        stock_code: 종목코드 (예: "005930")
        action: "buy" 또는 "sell"
        qty: 주문 수량
        price: 주문 가격
    """
    now = datetime.datetime.now()
    row = [
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        strategy_id,
        stock_code,
        action,
        qty,
        price,
        qty * price,
    ]
    with _lock:
        _ensure_csv(ORDERS_CSV, ORDERS_HEADER)
        with open(ORDERS_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    logger.info(f"[TradeLog] {strategy_id} | {action} {stock_code} x{qty} @{price}")


def log_daily_return(strategy_id: str, portfolio_value: float,
                     daily_pnl: float, daily_return_pct: float,
                     date: str | None = None):
    """
    일별 수익률을 CSV에 기록.

    Args:
        strategy_id: 전략 ID
        portfolio_value: 당일 장 마감 기준 포트폴리오 평가액
        daily_pnl: 당일 손익 (원)
        daily_return_pct: 당일 수익률 (%, 예: 0.48 → 0.48%)
        date: 날짜 (None이면 오늘)
    """
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    row = [date, strategy_id, portfolio_value, daily_pnl, daily_return_pct]
    with _lock:
        _ensure_csv(DAILY_RETURNS_CSV, DAILY_HEADER)
        with open(DAILY_RETURNS_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    logger.info(f"[DailyReturn] {strategy_id} | {date} | PnL={daily_pnl:+,.0f} | 수익률={daily_return_pct:+.2f}%")
