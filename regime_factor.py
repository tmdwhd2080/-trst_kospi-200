# -*- coding: utf-8 -*-
"""
27개 국면 조합별 팩터 수익률 분석 (독립 실행 버전)

3개 국면 × 3개 상태 = 27개 조합
- DRAI: Risk-On(1), Neutral(0), Risk-Off(-1)
- MACRO_GROWTH: Expansion(1), Neutral(0), Contraction(-1)
- MACRO_INFLATION: High(1), Moderate(0), Low(-1)

각 조합별로:
1. 해당 조합에 속하는 기간(주) 수
2. 각 팩터의 평균 수익률 (주간, 연율화)
"""

import os
import sys
import pymssql
import pandas as pd
import numpy as np
import logging
from itertools import product
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================================================================
# DB 설정 (variables.py 내용)
# =============================================================================
class DBConfig:
    MSSQL_SERVER = "192.168.50.52"
    MSSQL_USER = "trstdev"
    MSSQL_PASSWORD = "trst002!"
    TRSTDEV_DB = "TRSTDEV"


# =============================================================================
# 팩터/국면 매핑
# =============================================================================
FACTOR_MAPPING = {
    'CP_V': 'Value',
    'CP_G': 'Growth',
    'CP_Q': 'Quality',
    'CP_LV': 'LowVol',
    'CP_MOM': 'Momentum',
    'CP_S': 'Size',
}

FACTOR_CODE_MAPPING = {v: k for k, v in FACTOR_MAPPING.items()}

REGIME_CODES = {
    'RG00101': 'DRAI',
    'RG00211': 'MACRO_GROWTH',
    'RG00311': 'MACRO_INFLATION'
}

STATE_MAPPING = {
    1: 'Pos',
    0: 'Neu',
    -1: 'Neg'
}

REGIME_STATE_DESC = {
    'DRAI': {1: 'RiskOn', 0: 'Neutral', -1: 'RiskOff'},
    'MACRO_GROWTH': {1: 'Expand', 0: 'Neutral', -1: 'Contract'},
    'MACRO_INFLATION': {1: 'High', 0: 'Moderate', -1: 'Low'}
}

FACTOR_ORDER = ['Value', 'Growth', 'Quality', 'LowVol', 'Momentum', 'Size']


# =============================================================================
# MSSQL 클래스 (database2.py 내용)
# =============================================================================
class MSSQL:
    def __init__(self,
                 server: str = DBConfig.MSSQL_SERVER,
                 user: str = DBConfig.MSSQL_USER,
                 password: str = DBConfig.MSSQL_PASSWORD,
                 database: str = DBConfig.TRSTDEV_DB):

        self.server = server
        self.user = user
        self.password = password
        self.database = database
        self._conn = None
        self._cursor = None
        self._connect()

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def _connect(self):
        try:
            self._conn = pymssql.connect(
                server=self.server,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8'
            )
            self._cursor = self._conn.cursor()
            logging.info(f"Connected to {self.database}")

        except Exception as e:
            logging.error(f"Connection failed: {e}")
            raise

    def close(self):
        if self._conn:
            self._conn.close()

    def reconnect(self):
        self.close()
        self._connect()

    def SELECT(self,
               query: str,
               print_query: bool = False,
               output_type: str = "DataFrame",
               index_col: str = None) -> Optional[pd.DataFrame]:
        
        if output_type != "DataFrame" and index_col is not None:
            raise TypeError("index_col을 사용하려면 output_type이 'DataFrame'이어야 합니다.")

        try:
            self._cursor.execute(query)
            row = self._cursor.fetchone()

            header_list = [desc[0] for desc in self._cursor.description]

            if print_query:
                print('\t'.join(header_list))

            data_mat = []
            while row:
                row = list(row)
                row_list = []
                for r in row:
                    try:
                        if isinstance(r, str):
                            r = r.encode('ISO-8859-1').decode('euc-kr')
                    except:
                        pass
                    row_list.append(r)

                if print_query:
                    print('\t'.join(str(x) for x in row_list))

                data_mat.append(row_list)
                row = self._cursor.fetchone()

            if len(data_mat) == 0:
                return None

            if output_type == "old_version":
                return data_mat

            data_df = pd.DataFrame(data_mat, columns=header_list)

            if output_type == "DataFrame":
                if index_col is not None:
                    if index_col not in data_df.columns:
                        raise ValueError(f"'{index_col}' 컬럼이 존재하지 않습니다.")
                    data_df = data_df.set_index(index_col)

                if len(data_df.columns) == 1:
                    return data_df.iloc[:, 0]
                return data_df

            elif output_type == "array":
                return np.array(data_mat).T

            elif output_type == "list":
                return list(np.array(data_mat).T)

            else:
                raise ValueError(f"잘못된 output_type: {output_type}")

        except Exception as e:
            logging.error(f"SELECT 실패: {e}")
            raise


# =============================================================================
# 분석 설정
# =============================================================================
START_DATE = '2003-01-01'
END_DATE = '2025-12-19'


# =============================================================================
# 데이터 로드 함수
# =============================================================================
def load_factor_returns() -> pd.DataFrame:
    """DB에서 팩터 수익률 로드"""
    
    print("=" * 80)
    print("[Step 1] 팩터 수익률 데이터 로드")
    print("=" * 80)
    
    db = MSSQL()
    
    query = f"""
    SELECT 
        BaseDate,
        FLD_NAME,
        Rtn_L_S
    FROM PFM_FCTR
    WHERE MODEL = 'COM_FCTR'
      AND FREQ = 'W'
      AND LAG = 1
      AND BaseDate BETWEEN '{START_DATE}' AND '{END_DATE}'
      AND FLD_NAME IN ('CP_V', 'CP_G', 'CP_Q', 'CP_LV', 'CP_MOM', 'CP_S')
    ORDER BY BaseDate, FLD_NAME
    """
    
    df = db.SELECT(query)
    db.close()
    
    if df is None or len(df) == 0:
        raise ValueError("팩터 수익률 데이터가 없습니다.")
    
    df['BaseDate'] = pd.to_datetime(df['BaseDate'])
    df['Rtn_L_S'] = pd.to_numeric(df['Rtn_L_S'], errors='coerce')
    
    df_pivot = df.pivot_table(
        index='BaseDate',
        columns='FLD_NAME',
        values='Rtn_L_S',
        aggfunc='first'
    )
    
    df_pivot = df_pivot.rename(columns=FACTOR_MAPPING)
    df_pivot = df_pivot.sort_index()
    df_pivot = df_pivot.dropna(how='all')
    
    print(f"  기간: {df_pivot.index.min().date()} ~ {df_pivot.index.max().date()}")
    print(f"  관측 수: {len(df_pivot)}주")
    
    return df_pivot


def load_regime_data() -> pd.DataFrame:
    """DB에서 국면 데이터 로드"""
    
    print("\n" + "=" * 80)
    print("[Step 2] 국면 데이터 로드")
    print("=" * 80)
    
    db = MSSQL()
    
    query = f"""
    SELECT 
        LookBackDate,
        RegimeCode,
        STATES
    FROM REGIME_QMS
    WHERE RegimeCode IN ('RG00101', 'RG00211', 'RG00311')
      AND LookBackDate BETWEEN '{START_DATE}' AND '{END_DATE}'
      AND RECENT = 1
    ORDER BY LookBackDate, RegimeCode
    """
    
    df = db.SELECT(query)
    db.close()
    
    if df is None or len(df) == 0:
        raise ValueError("국면 데이터가 없습니다.")
    
    df['LookBackDate'] = pd.to_datetime(df['LookBackDate'])
    df['STATES'] = pd.to_numeric(df['STATES'], errors='coerce').astype(int)
    df['RegimeName'] = df['RegimeCode'].map(REGIME_CODES)
    
    df_pivot = df.pivot_table(
        index='LookBackDate',
        columns='RegimeName',
        values='STATES',
        aggfunc='first'
    )
    
    df_pivot = df_pivot.sort_index()
    
    print(f"  기간: {df_pivot.index.min().date()} ~ {df_pivot.index.max().date()}")
    print(f"  관측 수: {len(df_pivot)}")
    
    return df_pivot


def merge_data(factor_returns: pd.DataFrame, regime_data: pd.DataFrame) -> pd.DataFrame:
    """데이터 병합"""
    
    print("\n" + "=" * 80)
    print("[Step 3] 데이터 병합")
    print("=" * 80)
    
    regime_weekly = regime_data.resample('W-FRI').last()
    
    factor_returns.index.name = 'Date'
    regime_weekly.index.name = 'Date'
    
    merged = factor_returns.join(regime_weekly, how='inner')
    merged = merged.dropna()
    
    print(f"  병합 후 관측 수: {len(merged)}주")
    print(f"  기간: {merged.index.min().date()} ~ {merged.index.max().date()}")
    
    return merged


# =============================================================================
# 27개 조합별 분석
# =============================================================================
def analyze_all_combinations(merged_data: pd.DataFrame) -> pd.DataFrame:
    """27개 국면 조합별 팩터 수익률 분석"""
    
    print("\n" + "=" * 80)
    print("[Step 4] 27개 국면 조합별 분석")
    print("=" * 80)
    
    factor_cols = [c for c in FACTOR_ORDER if c in merged_data.columns]
    states = [1, 0, -1]
    
    results = []
    
    # 3^3 = 27개 조합
    for drai_state, growth_state, inflation_state in product(states, states, states):
        
        # 조합 마스크
        mask = (
            (merged_data['DRAI'] == drai_state) &
            (merged_data['MACRO_GROWTH'] == growth_state) &
            (merged_data['MACRO_INFLATION'] == inflation_state)
        )
        
        subset = merged_data.loc[mask, factor_cols]
        n_weeks = len(subset)
        
        # 조합 설명
        drai_desc = REGIME_STATE_DESC['DRAI'][drai_state]
        growth_desc = REGIME_STATE_DESC['MACRO_GROWTH'][growth_state]
        inflation_desc = REGIME_STATE_DESC['MACRO_INFLATION'][inflation_state]
        
        result = {
            'DRAI': drai_state,
            'DRAI_Desc': drai_desc,
            'GROWTH': growth_state,
            'GROWTH_Desc': growth_desc,
            'INFLATION': inflation_state,
            'INFLATION_Desc': inflation_desc,
            'N_Weeks': n_weeks,
        }
        
        # 각 팩터별 수익률
        for factor in factor_cols:
            if n_weeks > 0:
                weekly_ret = subset[factor].mean()
                annual_ret = weekly_ret * 52
                result[f'{factor}_Weekly'] = weekly_ret
                result[f'{factor}_Annual'] = annual_ret
            else:
                result[f'{factor}_Weekly'] = np.nan
                result[f'{factor}_Annual'] = np.nan
        
        results.append(result)
    
    return pd.DataFrame(results)


# =============================================================================
# 결과 출력 함수
# =============================================================================
def print_results(results_df: pd.DataFrame):
    """결과 출력"""
    
    print("\n" + "=" * 140)
    print("[결과] 27개 국면 조합별 팩터 수익률 (연율화)")
    print("=" * 140)
    
    factor_cols = [c for c in FACTOR_ORDER if f'{c}_Annual' in results_df.columns]
    
    # 헤더 출력
    header = f"{'DRAI':<8} {'GROWTH':<10} {'INFLATION':<10} {'N_Weeks':>8} |"
    for factor in factor_cols:
        header += f" {factor:>10}"
    print(header)
    print("-" * 140)
    
    # 데이터 출력 (N_Weeks 기준 내림차순 정렬)
    sorted_df = results_df.sort_values('N_Weeks', ascending=False)
    
    for _, row in sorted_df.iterrows():
        line = f"{row['DRAI_Desc']:<8} {row['GROWTH_Desc']:<10} {row['INFLATION_Desc']:<10} {row['N_Weeks']:>8} |"
        
        for factor in factor_cols:
            annual_ret = row[f'{factor}_Annual']
            if pd.notna(annual_ret):
                line += f" {annual_ret:>9.2%}"
            else:
                line += f" {'-':>10}"
        
        print(line)
    
    print("-" * 140)
    
    # 요약 통계
    print("\n" + "=" * 80)
    print("[요약 통계]")
    print("=" * 80)
    
    total_weeks = results_df['N_Weeks'].sum()
    non_zero = results_df[results_df['N_Weeks'] > 0]
    
    print(f"  전체 관측 주: {total_weeks}")
    print(f"  데이터 있는 조합: {len(non_zero)}/27개")
    print(f"  데이터 없는 조합: {27 - len(non_zero)}개")
    
    # 상위 5개 조합
    print("\n[관측 수 상위 5개 조합]")
    top5 = results_df.nlargest(5, 'N_Weeks')
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        pct = row['N_Weeks'] / total_weeks * 100 if total_weeks > 0 else 0
        print(f"  {i}. {row['DRAI_Desc']}/{row['GROWTH_Desc']}/{row['INFLATION_Desc']}: "
              f"{row['N_Weeks']}주 ({pct:.1f}%)")
    
    # 팩터별 최고/최저 수익률 조합
    print("\n[팩터별 최고 수익률 조합 (최소 10주 이상)]")
    for factor in factor_cols:
        col = f'{factor}_Annual'
        valid = results_df[results_df['N_Weeks'] >= 10]
        if len(valid) > 0 and valid[col].notna().any():
            best_idx = valid[col].idxmax()
            best = valid.loc[best_idx]
            print(f"  {factor:10s}: {best['DRAI_Desc']}/{best['GROWTH_Desc']}/{best['INFLATION_Desc']} "
                  f"= {best[col]:.2%} ({best['N_Weeks']}주)")
    
    print("\n[팩터별 최저 수익률 조합 (최소 10주 이상)]")
    for factor in factor_cols:
        col = f'{factor}_Annual'
        valid = results_df[results_df['N_Weeks'] >= 10]
        if len(valid) > 0 and valid[col].notna().any():
            worst_idx = valid[col].idxmin()
            worst = valid.loc[worst_idx]
            print(f"  {factor:10s}: {worst['DRAI_Desc']}/{worst['GROWTH_Desc']}/{worst['INFLATION_Desc']} "
                  f"= {worst[col]:.2%} ({worst['N_Weeks']}주)")


def print_heatmap_style(results_df: pd.DataFrame):
    """히트맵 스타일 출력 (DRAI x GROWTH, INFLATION 별로)"""
    
    print("\n" + "=" * 100)
    print("[히트맵 스타일 출력] - 관측 수 (주)")
    print("=" * 100)
    
    header_label = "DRAI \\ GROWTH"
    
    for inflation_state in [1, 0, -1]:
        inflation_desc = REGIME_STATE_DESC['MACRO_INFLATION'][inflation_state]
        print(f"\n  INFLATION = {inflation_desc}")
        print(f"  {header_label:<15}", end="")
        
        for growth_state in [1, 0, -1]:
            growth_desc = REGIME_STATE_DESC['MACRO_GROWTH'][growth_state]
            print(f"{growth_desc:>12}", end="")
        print("      Total")
        print("  " + "-" * 60)
        
        row_totals = []
        for drai_state in [1, 0, -1]:
            drai_desc = REGIME_STATE_DESC['DRAI'][drai_state]
            print(f"  {drai_desc:<15}", end="")
            
            row_sum = 0
            for growth_state in [1, 0, -1]:
                mask = (
                    (results_df['DRAI'] == drai_state) &
                    (results_df['GROWTH'] == growth_state) &
                    (results_df['INFLATION'] == inflation_state)
                )
                row = results_df[mask]
                
                if len(row) > 0:
                    n_weeks = row['N_Weeks'].values[0]
                    row_sum += n_weeks
                    print(f"{n_weeks:>12}", end="")
                else:
                    print(f"{'0':>12}", end="")
            
            print(f"{row_sum:>10}")
            row_totals.append(row_sum)
        
        # 열 합계
        print("  " + "-" * 60)
        print(f"  {'Total':<15}", end="")
        col_total = 0
        for growth_state in [1, 0, -1]:
            col_sum = 0
            for drai_state in [1, 0, -1]:
                mask = (
                    (results_df['DRAI'] == drai_state) &
                    (results_df['GROWTH'] == growth_state) &
                    (results_df['INFLATION'] == inflation_state)
                )
                row = results_df[mask]
                if len(row) > 0:
                    col_sum += row['N_Weeks'].values[0]
            print(f"{col_sum:>12}", end="")
            col_total += col_sum
        print(f"{col_total:>10}")


def print_pivot_table(results_df: pd.DataFrame, factor: str):
    """특정 팩터에 대한 피벗 테이블 출력"""
    
    print(f"\n{'=' * 80}")
    print(f"[{factor} 팩터 연율 수익률 - 피벗 테이블]")
    print(f"{'=' * 80}")
    
    col = f'{factor}_Annual'
    
    if col not in results_df.columns:
        print(f"  {factor} 팩터 데이터가 없습니다.")
        return
    
    # DRAI x GROWTH 피벗 (각 INFLATION 별로)
    for inflation_state in [1, 0, -1]:
        inflation_desc = REGIME_STATE_DESC['MACRO_INFLATION'][inflation_state]
        print(f"\n  INFLATION = {inflation_desc}")
        print(f"  {'':<12}", end="")
        
        for growth_state in [1, 0, -1]:
            growth_desc = REGIME_STATE_DESC['MACRO_GROWTH'][growth_state]
            print(f"{growth_desc:>15}", end="")
        print()
        
        for drai_state in [1, 0, -1]:
            drai_desc = REGIME_STATE_DESC['DRAI'][drai_state]
            print(f"  {drai_desc:<12}", end="")
            
            for growth_state in [1, 0, -1]:
                mask = (
                    (results_df['DRAI'] == drai_state) &
                    (results_df['GROWTH'] == growth_state) &
                    (results_df['INFLATION'] == inflation_state)
                )
                row = results_df[mask]
                
                if len(row) > 0 and pd.notna(row[col].values[0]):
                    val = row[col].values[0]
                    n_weeks = row['N_Weeks'].values[0]
                    print(f"{val:>9.2%}({n_weeks:>3})", end="")
                else:
                    print(f"{'--':>15}", end="")
            print()


# =============================================================================
# 메인 실행
# =============================================================================
if __name__ == '__main__':
    
    print("\n" + "=" * 80)
    print("27개 국면 조합별 팩터 수익률 분석")
    print("=" * 80)
    print("  조합: DRAI(3) × GROWTH(3) × INFLATION(3) = 27개")
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    
    try:
        # 1. 데이터 로드
        factor_returns = load_factor_returns()
        regime_data = load_regime_data()
        
        # 2. 데이터 병합
        merged_data = merge_data(factor_returns, regime_data)
        
        # 3. 27개 조합별 분석
        results_df = analyze_all_combinations(merged_data)
        
        # 4. 결과 출력
        print_results(results_df)
        
        # 5. 히트맵 스타일 출력
        print_heatmap_style(results_df)
        
        # 6. 특정 팩터 피벗 테이블 (선택적)
        print_pivot_table(results_df, 'Value')
        print_pivot_table(results_df, 'Momentum')
        
        print("\n" + "=" * 80)
        print("분석 완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()