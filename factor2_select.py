# -*- coding: utf-8 -*-
"""
2개 국면 조합별 팩터 순위(Rank) 빈도 분석

2개 국면 × 3개 상태 = 3^2 = 9가지 조합
(단, 실제로는 -1, 0, 1 세 상태이므로 9가지)

선택 가능한 2개 국면 조합 (3C2 = 3가지):
1. DRAI + MACRO_GROWTH
2. DRAI + MACRO_INFLATION
3. MACRO_GROWTH + MACRO_INFLATION

상단의 SELECTED_REGIMES 변수를 수정하여 원하는 조합 선택
"""

import pandas as pd
import numpy as np
from itertools import product
from collections import defaultdict

# 기존 코드에서 import
from regime_factor import (
    MSSQL, DBConfig, FACTOR_MAPPING, FACTOR_ORDER,
    REGIME_CODES, REGIME_STATE_DESC, STATE_MAPPING,
    START_DATE, END_DATE,
    load_factor_returns, load_regime_data, merge_data
)


# =============================================================================
# ★★★ 여기서 분석할 2개 국면 선택 ★★★
# =============================================================================
# 선택 옵션:
#   1) ['DRAI', 'MACRO_GROWTH']
#   2) ['DRAI', 'MACRO_INFLATION']
#   3) ['MACRO_GROWTH', 'MACRO_INFLATION']

SELECTED_REGIMES = ['MACRO_GROWTH', 'DRAI']  # ← 여기를 수정하세요!

# =============================================================================


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


def analyze_rank_frequency_by_two_regimes(ranks_data: pd.DataFrame, 
                                           regime1: str, 
                                           regime2: str) -> dict:
    """
    2개 국면 조합별 팩터 순위 빈도 분석
    3 × 3 = 9가지 조합
    """
    
    factor_cols = [c for c in FACTOR_ORDER if c in ranks_data.columns]
    states = [1, 0, -1]
    
    results = {}
    
    for state1, state2 in product(states, states):
        
        # 조합 마스크
        mask = (
            (ranks_data[regime1] == state1) &
            (ranks_data[regime2] == state2)
        )
        
        subset = ranks_data.loc[mask, factor_cols]
        n_weeks = len(subset)
        
        if n_weeks == 0:
            continue
        
        # 조합 키
        combo_key = (state1, state2)
        
        # 각 팩터별 순위 빈도 집계
        rank_freq = {}
        for factor in factor_cols:
            freq = subset[factor].value_counts().sort_index()
            rank_freq[factor] = {rank: freq.get(rank, 0) for rank in range(1, 7)}
        
        results[combo_key] = {
            'n_weeks': n_weeks,
            'rank_freq': rank_freq,
            'state1_desc': REGIME_STATE_DESC[regime1][state1],
            'state2_desc': REGIME_STATE_DESC[regime2][state2],
        }
    
    return results


def print_two_regime_rank_frequency(results: dict, regime1: str, regime2: str):
    """2개 국면 조합별 순위 빈도 출력"""
    
    print("\n" + "=" * 120)
    print(f"[결과] {regime1} × {regime2} 조합별 팩터 순위 빈도 (횟수)")
    print("=" * 120)
    print("  * 순위: 1등 = 해당 주에 가장 높은 수익률, 6등 = 가장 낮은 수익률")
    print(f"  * 조합 수: {len(results)}개")
    
    # N_Weeks 기준 내림차순 정렬
    sorted_combos = sorted(results.items(), key=lambda x: x[1]['n_weeks'], reverse=True)
    
    for combo_key, data in sorted_combos:
        state1_desc = data['state1_desc']
        state2_desc = data['state2_desc']
        n_weeks = data['n_weeks']
        rank_freq = data['rank_freq']
        
        print("\n" + "-" * 120)
        print(f"[{regime1}={state1_desc} / {regime2}={state2_desc}] - 총 {n_weeks}주")
        print("-" * 120)
        
        # 헤더
        header = f"{'Factor':<12} |"
        for rank in range(1, 7):
            header += f" {'Rank'+str(rank):>8}"
        header += " |  평균순위  1등비율  6등비율"
        print(header)
        print("-" * 120)
        
        for factor in FACTOR_ORDER:
            if factor not in rank_freq:
                continue
                
            freq = rank_freq[factor]
            line = f"{factor:<12} |"
            
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


def print_two_regime_rank_percentage(results: dict, regime1: str, regime2: str):
    """2개 국면 조합별 순위 빈도 (비율 %) 출력"""
    
    print("\n" + "=" * 120)
    print(f"[결과] {regime1} × {regime2} 조합별 팩터 순위 빈도 (비율 %)")
    print("=" * 120)
    
    sorted_combos = sorted(results.items(), key=lambda x: x[1]['n_weeks'], reverse=True)
    
    for combo_key, data in sorted_combos:
        state1_desc = data['state1_desc']
        state2_desc = data['state2_desc']
        n_weeks = data['n_weeks']
        rank_freq = data['rank_freq']
        
        if n_weeks < 10:
            continue
        
        print(f"\n  [{regime1}={state1_desc} / {regime2}={state2_desc}] - 총 {n_weeks}주")
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


def print_two_regime_comparison_table(results: dict, regime1: str, regime2: str):
    """2개 국면 조합별 팩터 평균 순위 비교 테이블"""
    
    print("\n" + "=" * 120)
    print(f"[비교] {regime1} × {regime2} 조합별 팩터 평균 순위")
    print("=" * 120)
    print("  * 낮을수록 해당 조합에서 성과가 좋음 (1.0 = 항상 1등)")
    
    # 유효한 조합만 (10주 이상)
    valid_combos = [(k, v) for k, v in results.items() if v['n_weeks'] >= 10]
    valid_combos = sorted(valid_combos, key=lambda x: x[1]['n_weeks'], reverse=True)
    
    if not valid_combos:
        print("  충분한 데이터가 있는 조합이 없습니다.")
        return
    
    # 헤더
    print(f"\n  {'Factor':<12}", end="")
    for combo_key, data in valid_combos:
        short_name = f"{data['state1_desc'][:3]}/{data['state2_desc'][:3]}"
        print(f" {short_name:>10}", end="")
    print("  |  최대-최소")
    
    print(f"  {'(N_Weeks)':<12}", end="")
    for combo_key, data in valid_combos:
        print(f" {data['n_weeks']:>10}", end="")
    print()
    print("  " + "-" * (14 + 11 * len(valid_combos) + 12))
    
    # 각 팩터별
    for factor in FACTOR_ORDER:
        print(f"  {factor:<12}", end="")
        
        avg_ranks = []
        for combo_key, data in valid_combos:
            rank_freq = data['rank_freq']
            n_weeks = data['n_weeks']
            
            if factor in rank_freq:
                freq = rank_freq[factor]
                total_rank = sum(rank * count for rank, count in freq.items())
                avg_rank = total_rank / n_weeks if n_weeks > 0 else 0
                avg_ranks.append(avg_rank)
                print(f" {avg_rank:>10.2f}", end="")
            else:
                print(f" {'-':>10}", end="")
        
        # 최대-최소 차이
        if len(avg_ranks) >= 2:
            diff = max(avg_ranks) - min(avg_ranks)
            print(f"  |    {diff:>5.2f}")
        else:
            print("  |      -")


def print_two_regime_best_worst(results: dict, regime1: str, regime2: str):
    """2개 국면 조합별 Best/Worst 팩터"""
    
    print("\n" + "=" * 120)
    print(f"[분석] {regime1} × {regime2} 조합별 Best/Worst 팩터")
    print("=" * 120)
    
    r1_short = regime1[:6]
    r2_short = regime2[:6]
    
    print(f"\n  {r1_short:<10} {r2_short:<10} {'N':>6} | {'Best 팩터':<12} {'평균순위':>8} | {'Worst 팩터':<12} {'평균순위':>8}")
    print("  " + "-" * 100)
    
    sorted_combos = sorted(results.items(), key=lambda x: x[1]['n_weeks'], reverse=True)
    
    for combo_key, data in sorted_combos:
        n_weeks = data['n_weeks']
        if n_weeks < 10:
            continue
            
        state1_desc = data['state1_desc']
        state2_desc = data['state2_desc']
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
            
            print(f"  {state1_desc:<10} {state2_desc:<10} {n_weeks:>6} | "
                  f"{best_factor[0]:<12} {best_factor[1]:>8.2f} | "
                  f"{worst_factor[0]:<12} {worst_factor[1]:>8.2f}")


def print_two_regime_heatmap(results: dict, regime1: str, regime2: str):
    """2개 국면 조합 히트맵 (3x3 매트릭스 형태)"""
    
    print("\n" + "=" * 120)
    print(f"[히트맵] {regime1} × {regime2} - 관측 수 (주)")
    print("=" * 120)
    
    states = [1, 0, -1]
    
    # 헤더
    r1_label = regime1 + " \\ " + regime2
    print(f"\n  {r1_label:<20}", end="")
    for state2 in states:
        state2_desc = REGIME_STATE_DESC[regime2][state2]
        print(f" {state2_desc:>12}", end="")
    print("      Total")
    print("  " + "-" * 70)
    
    # 각 행
    grand_total = 0
    for state1 in states:
        state1_desc = REGIME_STATE_DESC[regime1][state1]
        print(f"  {state1_desc:<20}", end="")
        
        row_sum = 0
        for state2 in states:
            combo_key = (state1, state2)
            if combo_key in results:
                n_weeks = results[combo_key]['n_weeks']
                row_sum += n_weeks
                print(f" {n_weeks:>12}", end="")
            else:
                print(f" {'0':>12}", end="")
        
        print(f" {row_sum:>10}")
        grand_total += row_sum
    
    # 열 합계
    print("  " + "-" * 70)
    print(f"  {'Total':<20}", end="")
    for state2 in states:
        col_sum = 0
        for state1 in states:
            combo_key = (state1, state2)
            if combo_key in results:
                col_sum += results[combo_key]['n_weeks']
        print(f" {col_sum:>12}", end="")
    print(f" {grand_total:>10}")


def print_factor_heatmap_by_two_regimes(results: dict, regime1: str, regime2: str, factor: str):
    """특정 팩터의 2개 국면 조합별 평균 순위 히트맵"""
    
    print(f"\n  [{factor}] 평균 순위 (3x3 매트릭스)")
    
    states = [1, 0, -1]
    
    # 헤더
    r1_label = regime1[:4] + " \\ " + regime2[:4]
    print(f"  {r1_label:<15}", end="")
    for state2 in states:
        state2_desc = REGIME_STATE_DESC[regime2][state2]
        print(f" {state2_desc:>10}", end="")
    print()
    print("  " + "-" * 50)
    
    # 각 행
    for state1 in states:
        state1_desc = REGIME_STATE_DESC[regime1][state1]
        print(f"  {state1_desc:<15}", end="")
        
        for state2 in states:
            combo_key = (state1, state2)
            if combo_key in results:
                data = results[combo_key]
                rank_freq = data['rank_freq']
                n_weeks = data['n_weeks']
                
                if factor in rank_freq and n_weeks >= 10:
                    freq = rank_freq[factor]
                    total_rank = sum(rank * count for rank, count in freq.items())
                    avg_rank = total_rank / n_weeks
                    print(f" {avg_rank:>10.2f}", end="")
                else:
                    print(f" {'-':>10}", end="")
            else:
                print(f" {'-':>10}", end="")
        print()


def print_all_factor_heatmaps(results: dict, regime1: str, regime2: str):
    """모든 팩터에 대한 히트맵 출력"""
    
    print("\n" + "=" * 120)
    print(f"[히트맵] {regime1} × {regime2} - 팩터별 평균 순위")
    print("=" * 120)
    print("  * 낮을수록 좋음 (1.0 = 항상 1등)")
    
    for factor in FACTOR_ORDER:
        print_factor_heatmap_by_two_regimes(results, regime1, regime2, factor)


# =============================================================================
# 메인 실행
# =============================================================================
if __name__ == '__main__':
    
    # 선택된 2개 국면 확인
    if len(SELECTED_REGIMES) != 2:
        raise ValueError("SELECTED_REGIMES는 정확히 2개의 국면을 포함해야 합니다.")
    
    regime1, regime2 = SELECTED_REGIMES
    
    valid_regimes = ['DRAI', 'MACRO_GROWTH', 'MACRO_INFLATION']
    if regime1 not in valid_regimes or regime2 not in valid_regimes:
        raise ValueError(f"유효하지 않은 국면입니다. 선택 가능: {valid_regimes}")
    
    print("\n" + "=" * 80)
    print("2개 국면 조합별 팩터 순위(Rank) 빈도 분석")
    print("=" * 80)
    print(f"  선택된 국면: {regime1} × {regime2}")
    print(f"  조합 수: 3 × 3 = 9가지")
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print("\n  ※ 다른 국면 조합을 보려면 코드 상단의 SELECTED_REGIMES를 수정하세요.")
    print("     옵션 1: ['DRAI', 'MACRO_GROWTH']")
    print("     옵션 2: ['DRAI', 'MACRO_INFLATION']")
    print("     옵션 3: ['MACRO_GROWTH', 'MACRO_INFLATION']")
    
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
        
        # 4. 2개 국면 조합별 순위 빈도 분석
        print("\n" + "=" * 80)
        print(f"[Step 5] {regime1} × {regime2} 조합별 순위 빈도 집계")
        print("=" * 80)
        results = analyze_rank_frequency_by_two_regimes(ranks_data, regime1, regime2)
        print(f"  분석된 조합 수: {len(results)}개")
        
        # 5. 결과 출력
        print_two_regime_rank_frequency(results, regime1, regime2)
        print_two_regime_rank_percentage(results, regime1, regime2)
        print_two_regime_comparison_table(results, regime1, regime2)
        print_two_regime_best_worst(results, regime1, regime2)
        print_two_regime_heatmap(results, regime1, regime2)
        print_all_factor_heatmaps(results, regime1, regime2)
        
        print("\n" + "=" * 80)
        print("분석 완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

# python factor2_select.py