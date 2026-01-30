# -*- coding: utf-8 -*-
"""
27개 국면 조합별 팩터 순위(Rank) 빈도 분석

각 주마다 6개 팩터의 수익률 순위(1등~6등)를 매기고,
27개 국면 조합별로 각 팩터가 몇 번 1등, 2등, ... 6등을 했는지 집계

목적: 국면별로 팩터 순위가 꾸준히 유지되는지 확인
"""

import pandas as pd
import numpy as np
from itertools import product
from collections import defaultdict

# 기존 코드에서 import (같은 폴더에 regime_combination_analysis_standalone.py 필요)
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


def analyze_rank_frequency_by_combination(ranks_data: pd.DataFrame) -> dict:
    """27개 국면 조합별 팩터 순위 빈도 분석"""
    
    factor_cols = [c for c in FACTOR_ORDER if c in ranks_data.columns]
    states = [1, 0, -1]
    
    results = {}
    
    for drai_state, growth_state, inflation_state in product(states, states, states):
        
        # 조합 마스크
        mask = (
            (ranks_data['DRAI'] == drai_state) &
            (ranks_data['MACRO_GROWTH'] == growth_state) &
            (ranks_data['MACRO_INFLATION'] == inflation_state)
        )
        
        subset = ranks_data.loc[mask, factor_cols]
        n_weeks = len(subset)
        
        if n_weeks == 0:
            continue
        
        # 조합 키
        combo_key = (drai_state, growth_state, inflation_state)
        
        # 각 팩터별 순위 빈도 집계
        rank_freq = {}
        for factor in factor_cols:
            freq = subset[factor].value_counts().sort_index()
            # 1~6등까지 모든 순위에 대해 빈도 저장
            rank_freq[factor] = {rank: freq.get(rank, 0) for rank in range(1, 7)}
        
        results[combo_key] = {
            'n_weeks': n_weeks,
            'rank_freq': rank_freq,
            'drai_desc': REGIME_STATE_DESC['DRAI'][drai_state],
            'growth_desc': REGIME_STATE_DESC['MACRO_GROWTH'][growth_state],
            'inflation_desc': REGIME_STATE_DESC['MACRO_INFLATION'][inflation_state],
        }
    
    return results


def print_rank_frequency_table(results: dict):
    """순위 빈도 테이블 출력"""
    
    print("\n" + "=" * 120)
    print("[결과] 27개 국면 조합별 팩터 순위 빈도")
    print("=" * 120)
    print("  * 순위: 1등 = 해당 주에 가장 높은 수익률, 6등 = 가장 낮은 수익률")
    print("  * 괄호 안 숫자: 해당 순위를 기록한 횟수")
    
    # 조합을 N_Weeks 기준 내림차순 정렬
    sorted_combos = sorted(results.items(), key=lambda x: x[1]['n_weeks'], reverse=True)
    
    for combo_key, data in sorted_combos:
        drai_desc = data['drai_desc']
        growth_desc = data['growth_desc']
        inflation_desc = data['inflation_desc']
        n_weeks = data['n_weeks']
        rank_freq = data['rank_freq']
        
        print("\n" + "-" * 120)
        print(f"[{drai_desc} / {growth_desc} / {inflation_desc}] - 총 {n_weeks}주")
        print("-" * 120)
        
        # 헤더
        header = f"{'Factor':<12} |"
        for rank in range(1, 7):
            header += f" {'Rank'+str(rank):>8}"
        header += " |  평균순위  1등비율"
        print(header)
        print("-" * 120)
        
        # 각 팩터별 출력
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
            
            # 평균 순위 계산
            avg_rank = total_rank / n_weeks if n_weeks > 0 else 0
            # 1등 비율
            first_rate = freq.get(1, 0) / n_weeks * 100 if n_weeks > 0 else 0
            
            line += f" |    {avg_rank:>5.2f}    {first_rate:>5.1f}%"
            print(line)


def print_rank_frequency_percentage(results: dict):
    """순위 빈도를 비율(%)로 출력"""
    
    print("\n" + "=" * 120)
    print("[결과] 27개 국면 조합별 팩터 순위 빈도 (비율 %)")
    print("=" * 120)
    
    sorted_combos = sorted(results.items(), key=lambda x: x[1]['n_weeks'], reverse=True)
    
    for combo_key, data in sorted_combos:
        drai_desc = data['drai_desc']
        growth_desc = data['growth_desc']
        inflation_desc = data['inflation_desc']
        n_weeks = data['n_weeks']
        rank_freq = data['rank_freq']
        
        if n_weeks < 10:  # 10주 미만은 스킵
            continue
        
        print("\n" + "-" * 100)
        print(f"[{drai_desc} / {growth_desc} / {inflation_desc}] - 총 {n_weeks}주")
        print("-" * 100)
        
        # 헤더
        header = f"{'Factor':<12} |"
        for rank in range(1, 7):
            header += f" {'R'+str(rank):>7}"
        header += " | 평균순위"
        print(header)
        print("-" * 100)
        
        for factor in FACTOR_ORDER:
            if factor not in rank_freq:
                continue
                
            freq = rank_freq[factor]
            line = f"{factor:<12} |"
            
            total_rank = 0
            for rank in range(1, 7):
                count = freq.get(rank, 0)
                pct = count / n_weeks * 100 if n_weeks > 0 else 0
                line += f" {pct:>6.1f}%"
                total_rank += rank * count
            
            avg_rank = total_rank / n_weeks if n_weeks > 0 else 0
            line += f" |   {avg_rank:>5.2f}"
            print(line)


def print_factor_consistency_analysis(results: dict):
    """팩터별 순위 일관성 분석"""
    
    print("\n" + "=" * 120)
    print("[분석] 팩터별 순위 일관성 (국면별 평균 순위)")
    print("=" * 120)
    print("  * 값이 낮을수록 해당 국면에서 성과가 좋음")
    print("  * 표준편차가 작을수록 순위가 일관됨")
    
    # 각 팩터별로 국면별 평균 순위 수집
    factor_avg_ranks = {factor: {} for factor in FACTOR_ORDER}
    
    for combo_key, data in results.items():
        n_weeks = data['n_weeks']
        if n_weeks < 10:
            continue
            
        rank_freq = data['rank_freq']
        combo_name = f"{data['drai_desc'][:2]}/{data['growth_desc'][:2]}/{data['inflation_desc'][:2]}"
        
        for factor in FACTOR_ORDER:
            if factor in rank_freq:
                freq = rank_freq[factor]
                total_rank = sum(rank * count for rank, count in freq.items())
                avg_rank = total_rank / n_weeks
                factor_avg_ranks[factor][combo_name] = avg_rank
    
    # 팩터별 출력
    for factor in FACTOR_ORDER:
        if not factor_avg_ranks[factor]:
            continue
            
        ranks = factor_avg_ranks[factor]
        avg = np.mean(list(ranks.values()))
        std = np.std(list(ranks.values()))
        
        best_combo = min(ranks.items(), key=lambda x: x[1])
        worst_combo = max(ranks.items(), key=lambda x: x[1])
        
        print(f"\n  [{factor}]")
        print(f"    전체 평균 순위: {avg:.2f} (표준편차: {std:.2f})")
        print(f"    최고 성과 국면: {best_combo[0]} (평균 {best_combo[1]:.2f}등)")
        print(f"    최저 성과 국면: {worst_combo[0]} (평균 {worst_combo[1]:.2f}등)")


def print_top_bottom_analysis(results: dict):
    """각 국면에서 Top/Bottom 팩터 분석"""
    
    print("\n" + "=" * 120)
    print("[분석] 국면별 Best/Worst 팩터 (1등 비율 기준)")
    print("=" * 120)
    
    sorted_combos = sorted(results.items(), key=lambda x: x[1]['n_weeks'], reverse=True)
    
    print(f"\n{'국면 조합':<30} {'N':>6} | {'Best 팩터':<12} {'1등%':>7} | {'Worst 팩터':<12} {'6등%':>7}")
    print("-" * 100)
    
    for combo_key, data in sorted_combos:
        n_weeks = data['n_weeks']
        if n_weeks < 10:
            continue
            
        rank_freq = data['rank_freq']
        combo_name = f"{data['drai_desc']}/{data['growth_desc']}/{data['inflation_desc']}"
        
        # 1등 비율이 가장 높은 팩터
        best_factor = None
        best_rate = -1
        for factor, freq in rank_freq.items():
            rate = freq.get(1, 0) / n_weeks * 100
            if rate > best_rate:
                best_rate = rate
                best_factor = factor
        
        # 6등 비율이 가장 높은 팩터
        worst_factor = None
        worst_rate = -1
        for factor, freq in rank_freq.items():
            rate = freq.get(6, 0) / n_weeks * 100
            if rate > worst_rate:
                worst_rate = rate
                worst_factor = factor
        
        print(f"{combo_name:<30} {n_weeks:>6} | {best_factor:<12} {best_rate:>6.1f}% | {worst_factor:<12} {worst_rate:>6.1f}%")


def print_rank_stability_heatmap(results: dict):
    """순위 안정성 히트맵 (팩터 × 국면)"""
    
    print("\n" + "=" * 140)
    print("[히트맵] 팩터별 국면별 평균 순위")
    print("=" * 140)
    print("  * 낮을수록 좋음 (1.0 = 항상 1등, 6.0 = 항상 6등)")
    
    # 데이터가 충분한 조합만 선택
    valid_combos = [(k, v) for k, v in results.items() if v['n_weeks'] >= 10]
    valid_combos = sorted(valid_combos, key=lambda x: x[1]['n_weeks'], reverse=True)
    
    if not valid_combos:
        print("  충분한 데이터가 있는 조합이 없습니다.")
        return
    
    # 헤더 출력
    print(f"\n{'Factor':<12}", end="")
    for combo_key, data in valid_combos:
        short_name = f"{data['drai_desc'][:2]}/{data['growth_desc'][:2]}/{data['inflation_desc'][:2]}"
        print(f" {short_name:>10}", end="")
    print()
    print("-" * (12 + 11 * len(valid_combos)))
    
    # 각 팩터별 평균 순위 출력
    for factor in FACTOR_ORDER:
        print(f"{factor:<12}", end="")
        
        for combo_key, data in valid_combos:
            rank_freq = data['rank_freq']
            n_weeks = data['n_weeks']
            
            if factor in rank_freq:
                freq = rank_freq[factor]
                total_rank = sum(rank * count for rank, count in freq.items())
                avg_rank = total_rank / n_weeks
                print(f" {avg_rank:>10.2f}", end="")
            else:
                print(f" {'-':>10}", end="")
        print()


# =============================================================================
# 메인 실행
# =============================================================================
if __name__ == '__main__':
    
    print("\n" + "=" * 80)
    print("27개 국면 조합별 팩터 순위(Rank) 빈도 분석")
    print("=" * 80)
    print("  목적: 국면별로 팩터 순위가 꾸준히 유지되는지 확인")
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
        
        # 4. 국면 조합별 순위 빈도 분석
        print("\n" + "=" * 80)
        print("[Step 5] 국면 조합별 순위 빈도 집계")
        print("=" * 80)
        results = analyze_rank_frequency_by_combination(ranks_data)
        print(f"  분석된 조합 수: {len(results)}개")
        
        # 5. 결과 출력
        print_rank_frequency_table(results)
        print_rank_frequency_percentage(results)
        print_factor_consistency_analysis(results)
        print_top_bottom_analysis(results)
        print_rank_stability_heatmap(results)
        
        print("\n" + "=" * 80)
        print("분석 완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

# python factor_27count.py