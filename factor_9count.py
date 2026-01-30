# -*- coding: utf-8 -*-
"""
9가지 국면별 팩터 순위(Rank) 빈도 분석

3개 국면 × 3개 상태 = 9가지 (조합이 아닌 개별 분석)
- DRAI: Risk-On(1), Neutral(0), Risk-Off(-1) → 3가지
- MACRO_GROWTH: Expansion(1), Neutral(0), Contraction(-1) → 3가지
- MACRO_INFLATION: High(1), Moderate(0), Low(-1) → 3가지

각 국면 상태별로 해당 기간의 팩터 순위 빈도 집계
(예: DRAI가 Risk-On인 모든 주에서 각 팩터가 1등~6등을 몇 번 했는지)
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# 기존 코드에서 import
from regime_factor import (
    MSSQL, DBConfig, FACTOR_MAPPING, FACTOR_ORDER,
    REGIME_CODES, REGIME_STATE_DESC, STATE_MAPPING,
    START_DATE, END_DATE,
    load_factor_returns, load_regime_data, merge_data
)


def calculate_weekly_ranks(merged_data: pd.DataFrame) -> pd.DataFrame:
    """각 주별로 팩터 수익률 순위 계산 (1등이 가장 높은 수익률)"""
    
    factor_cols = [c for c in FACTOR_ORDER if c in merged_data.columns]
    
    # 각 주별 순위 계산 (ascending=False: 높은 수익률이 1등)
    ranks = merged_data[factor_cols].rank(axis=1, ascending=False, method='min')
    ranks = ranks.astype(int)
    
    # 국면 정보 추가
    for col in ['DRAI', 'MACRO_GROWTH', 'MACRO_INFLATION']:
        if col in merged_data.columns:
            ranks[col] = merged_data[col]
    
    return ranks


def analyze_rank_frequency_by_single_regime(ranks_data: pd.DataFrame) -> dict:
    """
    9가지 개별 국면별 팩터 순위 빈도 분석
    (27개 조합이 아닌 각 국면을 독립적으로 분석)
    """
    
    factor_cols = [c for c in FACTOR_ORDER if c in ranks_data.columns]
    states = [1, 0, -1]
    
    results = {}
    
    # 각 국면별로 개별 분석
    for regime in ['DRAI', 'MACRO_GROWTH', 'MACRO_INFLATION']:
        results[regime] = {}
        
        for state in states:
            # 해당 국면의 해당 상태인 모든 주 선택
            mask = ranks_data[regime] == state
            subset = ranks_data.loc[mask, factor_cols]
            n_weeks = len(subset)
            
            if n_weeks == 0:
                continue
            
            # 각 팩터별 순위 빈도 집계
            rank_freq = {}
            for factor in factor_cols:
                freq = subset[factor].value_counts().sort_index()
                rank_freq[factor] = {rank: freq.get(rank, 0) for rank in range(1, 7)}
            
            results[regime][state] = {
                'n_weeks': n_weeks,
                'rank_freq': rank_freq,
                'state_desc': REGIME_STATE_DESC[regime][state],
            }
    
    return results


def print_single_regime_rank_frequency(results: dict):
    """9가지 개별 국면별 순위 빈도 출력"""
    
    print("\n" + "=" * 120)
    print("[결과] 9가지 개별 국면별 팩터 순위 빈도 (횟수)")
    print("=" * 120)
    print("  * 순위: 1등 = 해당 주에 가장 높은 수익률, 6등 = 가장 낮은 수익률")
    print("  * 각 국면을 독립적으로 분석 (조합이 아님)")
    
    for regime in ['DRAI', 'MACRO_GROWTH', 'MACRO_INFLATION']:
        print("\n" + "=" * 120)
        print(f"[{regime}]")
        print("=" * 120)
        
        for state in [1, 0, -1]:
            if state not in results[regime]:
                continue
                
            data = results[regime][state]
            state_desc = data['state_desc']
            n_weeks = data['n_weeks']
            rank_freq = data['rank_freq']
            
            print(f"\n  [{state_desc}] - 총 {n_weeks}주")
            print("  " + "-" * 100)
            
            # 헤더
            header = f"  {'Factor':<12} |"
            for rank in range(1, 7):
                header += f" {'Rank'+str(rank):>8}"
            header += " |  평균순위  1등비율  6등비율"
            print(header)
            print("  " + "-" * 100)
            
            # 각 팩터별 출력
            for factor in FACTOR_ORDER:
                if factor not in rank_freq:
                    continue
                    
                freq = rank_freq[factor]
                line = f"  {factor:<12} |"
                
                total_rank = 0
                for rank in range(1, 7):
                    count = freq.get(rank, 0)
                    line += f" {count:>8}"
                    total_rank += rank * count
                
                avg_rank = total_rank / n_weeks if n_weeks > 0 else 0
                first_rate = freq.get(1, 0) / n_weeks * 100 if n_weeks > 0 else 0
                last_rate = freq.get(6, 0) / n_weeks * 100 if n_weeks > 0 else 0
                
                line += f" |    {avg_rank:>5.2f}   {first_rate:>5.1f}%   {last_rate:>5.1f}%"
                print(line)


def print_single_regime_rank_percentage(results: dict):
    """9가지 개별 국면별 순위 빈도 (비율 %) 출력"""
    
    print("\n" + "=" * 120)
    print("[결과] 9가지 개별 국면별 팩터 순위 빈도 (비율 %)")
    print("=" * 120)
    
    for regime in ['DRAI', 'MACRO_GROWTH', 'MACRO_INFLATION']:
        print("\n" + "=" * 120)
        print(f"[{regime}]")
        print("=" * 120)
        
        for state in [1, 0, -1]:
            if state not in results[regime]:
                continue
                
            data = results[regime][state]
            state_desc = data['state_desc']
            n_weeks = data['n_weeks']
            rank_freq = data['rank_freq']
            
            print(f"\n  [{state_desc}] - 총 {n_weeks}주")
            print("  " + "-" * 90)
            
            # 헤더
            header = f"  {'Factor':<12} |"
            for rank in range(1, 7):
                header += f"   R{rank:>1}"
            header += "  | 평균순위"
            print(header)
            print("  " + "-" * 90)
            
            for factor in FACTOR_ORDER:
                if factor not in rank_freq:
                    continue
                    
                freq = rank_freq[factor]
                line = f"  {factor:<12} |"
                
                total_rank = 0
                for rank in range(1, 7):
                    count = freq.get(rank, 0)
                    pct = count / n_weeks * 100 if n_weeks > 0 else 0
                    line += f" {pct:>4.0f}%"
                    total_rank += rank * count
                
                avg_rank = total_rank / n_weeks if n_weeks > 0 else 0
                line += f"  |   {avg_rank:>5.2f}"
                print(line)


def print_regime_comparison_table(results: dict):
    """국면별 팩터 평균 순위 비교 테이블"""
    
    print("\n" + "=" * 120)
    print("[비교] 국면별 팩터 평균 순위 비교")
    print("=" * 120)
    print("  * 낮을수록 해당 국면에서 성과가 좋음 (1.0 = 항상 1등)")
    
    for regime in ['DRAI', 'MACRO_GROWTH', 'MACRO_INFLATION']:
        print(f"\n  [{regime}]")
        print("  " + "-" * 80)
        
        # 헤더
        header = f"  {'Factor':<12} |"
        for state in [1, 0, -1]:
            if state in results[regime]:
                state_desc = results[regime][state]['state_desc']
                header += f" {state_desc:>12}"
        header += " |    차이"
        print(header)
        print("  " + "-" * 80)
        
        for factor in FACTOR_ORDER:
            line = f"  {factor:<12} |"
            avg_ranks = []
            
            for state in [1, 0, -1]:
                if state not in results[regime]:
                    line += f" {'-':>12}"
                    continue
                    
                data = results[regime][state]
                rank_freq = data['rank_freq']
                n_weeks = data['n_weeks']
                
                if factor in rank_freq:
                    freq = rank_freq[factor]
                    total_rank = sum(rank * count for rank, count in freq.items())
                    avg_rank = total_rank / n_weeks if n_weeks > 0 else 0
                    avg_ranks.append(avg_rank)
                    line += f" {avg_rank:>12.2f}"
                else:
                    line += f" {'-':>12}"
            
            # 최대-최소 차이
            if len(avg_ranks) >= 2:
                diff = max(avg_ranks) - min(avg_ranks)
                line += f" |   {diff:>5.2f}"
            else:
                line += " |     -"
            
            print(line)


def print_best_worst_by_regime(results: dict):
    """각 국면 상태별 Best/Worst 팩터"""
    
    print("\n" + "=" * 120)
    print("[분석] 각 국면 상태별 Best/Worst 팩터")
    print("=" * 120)
    
    print(f"\n  {'국면':<20} {'상태':<12} {'N':>6} | {'Best 팩터':<12} {'평균순위':>8} | {'Worst 팩터':<12} {'평균순위':>8}")
    print("  " + "-" * 100)
    
    for regime in ['DRAI', 'MACRO_GROWTH', 'MACRO_INFLATION']:
        for state in [1, 0, -1]:
            if state not in results[regime]:
                continue
                
            data = results[regime][state]
            state_desc = data['state_desc']
            n_weeks = data['n_weeks']
            rank_freq = data['rank_freq']
            
            # 평균 순위 계산
            avg_ranks = {}
            for factor in FACTOR_ORDER:
                if factor in rank_freq:
                    freq = rank_freq[factor]
                    total_rank = sum(rank * count for rank, count in freq.items())
                    avg_ranks[factor] = total_rank / n_weeks if n_weeks > 0 else 0
            
            if avg_ranks:
                best_factor = min(avg_ranks.items(), key=lambda x: x[1])
                worst_factor = max(avg_ranks.items(), key=lambda x: x[1])
                
                print(f"  {regime:<20} {state_desc:<12} {n_weeks:>6} | "
                      f"{best_factor[0]:<12} {best_factor[1]:>8.2f} | "
                      f"{worst_factor[0]:<12} {worst_factor[1]:>8.2f}")


def print_factor_regime_sensitivity(results: dict):
    """팩터별 국면 민감도 분석"""
    
    print("\n" + "=" * 120)
    print("[분석] 팩터별 국면 민감도 (순위 변동폭)")
    print("=" * 120)
    print("  * 변동폭이 클수록 국면에 따라 성과 차이가 큼")
    
    for factor in FACTOR_ORDER:
        print(f"\n  [{factor}]")
        
        for regime in ['DRAI', 'MACRO_GROWTH', 'MACRO_INFLATION']:
            avg_ranks = []
            state_info = []
            
            for state in [1, 0, -1]:
                if state not in results[regime]:
                    continue
                    
                data = results[regime][state]
                rank_freq = data['rank_freq']
                n_weeks = data['n_weeks']
                
                if factor in rank_freq:
                    freq = rank_freq[factor]
                    total_rank = sum(rank * count for rank, count in freq.items())
                    avg_rank = total_rank / n_weeks if n_weeks > 0 else 0
                    avg_ranks.append(avg_rank)
                    state_info.append((data['state_desc'], avg_rank))
            
            if len(avg_ranks) >= 2:
                diff = max(avg_ranks) - min(avg_ranks)
                best = min(state_info, key=lambda x: x[1])
                worst = max(state_info, key=lambda x: x[1])
                
                print(f"    {regime:<18}: 변동폭 {diff:.2f} "
                      f"(Best: {best[0]} {best[1]:.2f}, Worst: {worst[0]} {worst[1]:.2f})")


def print_summary_heatmap(results: dict):
    """요약 히트맵 (팩터 × 국면상태)"""
    
    print("\n" + "=" * 140)
    print("[요약 히트맵] 팩터별 국면 상태별 평균 순위")
    print("=" * 140)
    print("  * 낮을수록 좋음 (1.0 = 항상 1등, 6.0 = 항상 6등)")
    
    # 모든 국면 상태 리스트
    all_states = []
    for regime in ['DRAI', 'MACRO_GROWTH', 'MACRO_INFLATION']:
        for state in [1, 0, -1]:
            if state in results[regime]:
                state_desc = results[regime][state]['state_desc']
                n_weeks = results[regime][state]['n_weeks']
                all_states.append((regime, state, f"{regime[:4]}_{state_desc}", n_weeks))
    
    # 헤더
    print(f"\n  {'Factor':<12}", end="")
    for regime, state, label, n_weeks in all_states:
        print(f" {label:>12}", end="")
    print()
    
    print(f"  {'(N_Weeks)':<12}", end="")
    for regime, state, label, n_weeks in all_states:
        print(f" {n_weeks:>12}", end="")
    print()
    print("  " + "-" * (12 + 13 * len(all_states)))
    
    # 각 팩터별
    for factor in FACTOR_ORDER:
        print(f"  {factor:<12}", end="")
        
        for regime, state, label, n_weeks in all_states:
            data = results[regime][state]
            rank_freq = data['rank_freq']
            
            if factor in rank_freq:
                freq = rank_freq[factor]
                total_rank = sum(rank * count for rank, count in freq.items())
                avg_rank = total_rank / n_weeks if n_weeks > 0 else 0
                print(f" {avg_rank:>12.2f}", end="")
            else:
                print(f" {'-':>12}", end="")
        print()


# =============================================================================
# 메인 실행
# =============================================================================
if __name__ == '__main__':
    
    print("\n" + "=" * 80)
    print("9가지 개별 국면별 팩터 순위(Rank) 빈도 분석")
    print("=" * 80)
    print("  분석 대상: DRAI(3) + GROWTH(3) + INFLATION(3) = 9가지 개별 국면")
    print("  목적: 각 국면 상태별로 팩터 순위가 어떻게 분포하는지 확인")
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    
    try:
        # 1. 데이터 로드
        factor_returns = load_factor_returns()
        regime_data = load_regime_data()
        
        # 2. 데이터 병합
        merged_data = merge_data(factor_returns, regime_data)
        
        # 3. 주별 순위 계산
        print("\n" + "=" * 80)
        print("[Step 4] 주별 팩터 순위 계산")
        print("=" * 80)
        ranks_data = calculate_weekly_ranks(merged_data)
        print(f"  순위 계산 완료: {len(ranks_data)}주")
        
        # 4. 9가지 개별 국면별 순위 빈도 분석
        print("\n" + "=" * 80)
        print("[Step 5] 9가지 개별 국면별 순위 빈도 집계")
        print("=" * 80)
        results = analyze_rank_frequency_by_single_regime(ranks_data)
        total_states = sum(len(v) for v in results.values())
        print(f"  분석된 국면 상태 수: {total_states}개")
        
        # 5. 결과 출력
        print_single_regime_rank_frequency(results)
        print_single_regime_rank_percentage(results)
        print_regime_comparison_table(results)
        print_best_worst_by_regime(results)
        print_factor_regime_sensitivity(results)
        print_summary_heatmap(results)
        
        print("\n" + "=" * 80)
        print("분석 완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

# python factor_9count.py