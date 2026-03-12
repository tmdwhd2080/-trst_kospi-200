"""
전략 성과 분석기
특정 전략의 일별 수익률 로그를 읽어 MDD, 샤프 비율, 수익률을 계산합니다.
"""

import os
import csv
import math
import datetime

# 설정
STRATEGY_ID = "factor_momentum"                          # 분석할 전략 ID
STRATEGY_PATH = "strategies/factor_momentum_allocation.py"  # 전략 파일 경로 (참고용)
START_DATE = "2026-01-01"                                # 조회 시작일 (YYYY-MM-DD)
END_DATE = "2026-12-31"                                  # 조회 종료일 (YYYY-MM-DD)
RISK_FREE_RATE = 0.035                                   # 무위험 이자율 (연, 3.5%)
TRADING_DAYS_PER_YEAR = 252                              # 연간 거래일 수

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DAILY_RETURNS_CSV = os.path.join(_BASE_DIR, "data", "daily_returns.csv")


def load_daily_returns(strategy_id: str, start_date: str, end_date: str) -> list[dict]:
    if not os.path.exists(DAILY_RETURNS_CSV):
        print(f"❌ 데이터 파일이 없습니다: {DAILY_RETURNS_CSV}")
        print("   전략 실행 후 trade_logger로 일별 수익률을 기록해야 합니다.")
        return []

    records = []
    with open(DAILY_RETURNS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["strategy_id"] != strategy_id:
                continue
            if row["date"] < start_date or row["date"] > end_date:
                continue
            records.append({
                "date": row["date"],
                "portfolio_value": float(row["portfolio_value"]),
                "daily_pnl": float(row["daily_pnl"]),
                "daily_return_pct": float(row["daily_return_pct"]),
            })

    records.sort(key=lambda x: x["date"])
    return records


def calc_cumulative_return(records: list[dict]) -> float:
    """누적 수익률 (%) 계산"""
    if not records:
        return 0.0
    cumulative = 1.0
    for r in records:
        cumulative *= (1 + r["daily_return_pct"] / 100)
    return (cumulative - 1) * 100


def calc_annualized_return(cumulative_return_pct: float, trading_days: int) -> float:
    """연환산 수익률 (%) 계산."""
    if trading_days <= 0:
        return 0.0
    cumulative = 1 + cumulative_return_pct / 100
    if cumulative <= 0:
        return -100.0
    annualized = cumulative ** (TRADING_DAYS_PER_YEAR / trading_days) - 1
    return annualized * 100


def calc_mdd(records: list[dict]) -> tuple[float, str, str]:
    """
    MDD  계산.
    """
    if not records:
        return 0.0, "", ""

    equity = [1.0]
    for r in records:
        equity.append(equity[-1] * (1 + r["daily_return_pct"] / 100))

    peak = equity[0]
    peak_idx = 0
    max_dd = 0.0
    dd_peak_idx = 0
    dd_trough_idx = 0

    for i in range(1, len(equity)):
        if equity[i] > peak:
            peak = equity[i]
            peak_idx = i
        dd = (peak - equity[i]) / peak
        if dd > max_dd:
            max_dd = dd
            dd_peak_idx = peak_idx
            dd_trough_idx = i

    # 인덱스 → 날짜 매핑 
    peak_date = records[max(0, dd_peak_idx - 1)]["date"] if records else ""
    trough_date = records[min(dd_trough_idx - 1, len(records) - 1)]["date"] if records else ""

    return max_dd * 100, peak_date, trough_date


def calc_sharpe_ratio(records: list[dict]) -> float:
    """
    샤프 비율 계산.
    """
    if len(records) < 2:
        return 0.0

    daily_returns = [r["daily_return_pct"] / 100 for r in records]
    daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR

    # 평균
    n = len(daily_returns)
    mean_excess = sum(r - daily_rf for r in daily_returns) / n

    # 표준편차
    variance = sum((r - daily_rf - mean_excess) ** 2 for r in daily_returns) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0

    if std == 0:
        return 0.0

    return (mean_excess / std) * math.sqrt(TRADING_DAYS_PER_YEAR)


def calc_win_rate(records: list[dict]) -> tuple[float, int, int]:
    """승률 계산"""
    wins = sum(1 for r in records if r["daily_pnl"] > 0)
    losses = sum(1 for r in records if r["daily_pnl"] < 0)
    total = wins + losses
    rate = (wins / total * 100) if total > 0 else 0.0
    return rate, wins, losses


def analyze():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║         전략 성과 분석 (Performance Analyzer)     ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print(f"  전략 ID:    {STRATEGY_ID}")
    print(f"  전략 파일:  {STRATEGY_PATH}")
    print(f"  분석 기간:  {START_DATE} ~ {END_DATE}")
    print(f"  무위험이자: {RISK_FREE_RATE * 100:.1f}%")
    print()

    records = load_daily_returns(STRATEGY_ID, START_DATE, END_DATE)

    if not records:
        print("⚠ 해당 기간에 데이터가 없습니다.")
        print("  전략이 실행되면서 trade_logger.log_daily_return()으로")
        print("  일별 수익률을 기록해야 분석이 가능합니다.")
        return

    # 계산
    trading_days = len(records)
    cumulative_ret = calc_cumulative_return(records)
    annualized_ret = calc_annualized_return(cumulative_ret, trading_days)
    mdd, mdd_peak, mdd_trough = calc_mdd(records)
    sharpe = calc_sharpe_ratio(records)
    win_rate, wins, losses = calc_win_rate(records)

    # 일별 수익률 통계
    daily_rets = [r["daily_return_pct"] for r in records]
    avg_daily = sum(daily_rets) / len(daily_rets)
    max_daily = max(daily_rets)
    min_daily = min(daily_rets)
    total_pnl = sum(r["daily_pnl"] for r in records)

    last_portfolio = records[-1]["portfolio_value"]

    # 출력
    print("═" * 50)
    print("  📊 성과 요약")
    print("═" * 50)
    print(f"  거래일 수:        {trading_days}일")
    print(f"  최종 포트폴리오:  {last_portfolio:,.0f}원")
    print(f"  총 손익:          {total_pnl:+,.0f}원")
    print()
    print(f"  ▸ 누적 수익률:    {cumulative_ret:+.2f}%")
    print(f"  ▸ 연환산 수익률:  {annualized_ret:+.2f}%")
    print(f"  ▸ MDD:            {mdd:.2f}%  ({mdd_peak} ~ {mdd_trough})")
    print(f"  ▸ 샤프 비율:      {sharpe:.3f}")
    print()
    print(f"  ▸ 승률:           {win_rate:.1f}%  (이긴 {wins}일 / 진 {losses}일)")
    print(f"  ▸ 평균 일별 수익: {avg_daily:+.3f}%")
    print(f"  ▸ 최대 일 수익:   {max_daily:+.3f}%")
    print(f"  ▸ 최대 일 손실:   {min_daily:+.3f}%")
    print("═" * 50)


if __name__ == "__main__":
    analyze()
