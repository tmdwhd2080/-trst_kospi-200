# -*- coding: utf-8 -*-
"""
팩터별 AR(p) 자기상관 롤링 검증
- R_t = α + ρ₁·R_{t-1} + ρ₂·R_{t-2} + ... + ρₚ·R_{t-p} + ε
- lag(p)를 직접 설정 가능
- 롤링 윈도우 방식으로 전 기간 시행
- t_stat > 임계값 AND coef > 0 인 팩터를 시점별로 판별
"""

import pymssql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap

# ──────────────────────────────────────────────
# 1. DB 설정
# ──────────────────────────────────────────────
DB_CONFIG = {
    "server": "192.168.50.52",
    "user": "trstdev",
    "password": "trst002!",
    "database": "TRSTDEV",
}

FACTOR_MAPPING = {
    "CP_V": "Value",
    "CP_G": "Growth",
    "CP_Q": "Quality",
    "CP_LV": "LowVol",
    "CP_MOM": "Momentum",
    "CP_S": "Size",
}

COLORS = {
    "Value": "#1f77b4",
    "Growth": "#ff7f0e",
    "Quality": "#2ca02c",
    "LowVol": "#d62728",
    "Momentum": "#9467bd",
    "Size": "#8c564b",
}

FACTOR_ORDER = ["Value", "Growth", "Quality", "LowVol", "Momentum", "Size"]


# ──────────────────────────────────────────────
# 2. 데이터 조회
# ──────────────────────────────────────────────
def fetch_factor_data(start_date: str = "20040201", end_date: str = "20251219") -> pd.DataFrame:
    fld_names = "', '".join(FACTOR_MAPPING.keys())

    query = f"""
        SELECT BaseDate, FLD_NAME, LnRtn_L_S
        FROM PFM_FCTR
        WHERE BaseDate >= '{start_date}'
          AND BaseDate <= '{end_date}'
          AND FLD_NAME IN ('{fld_names}')
          AND FREQ = 'W'
          AND LAG = 1
          AND MODEL = 'COM_FCTR'
        ORDER BY BaseDate, FLD_NAME
    """

    conn = pymssql.connect(**DB_CONFIG, charset="utf8")
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()

    if df.empty:
        print("⚠️ 데이터가 없습니다.")
        return df

    for col in ["FLD_NAME"]:
        df[col] = df[col].apply(
            lambda x: x.encode("ISO-8859-1").decode("euc-kr") if isinstance(x, str) else x
        )

    df["BaseDate"] = pd.to_datetime(df["BaseDate"])
    df["LnRtn_L_S"] = pd.to_numeric(df["LnRtn_L_S"], errors="coerce")
    df["FactorName"] = df["FLD_NAME"].map(FACTOR_MAPPING)

    return df


# ──────────────────────────────────────────────
# 3. AR(p) 롤링 OLS (외부 라이브러리 없이 구현)
# ──────────────────────────────────────────────
def ols_ar(y: np.ndarray, lags: list):
    """
    OLS로 AR(p) 추정: R_t = α + ρ₁·R_{t-lag1} + ρ₂·R_{t-lag2} + ... + ε

    Parameters:
        y: 1D array (시계열)
        lags: lag 리스트 (예: [1] → AR(1), [1,2] → AR(2), [2] → 2주 전만)

    Returns:
        dict with keys per lag:
          {lag: {"coef": ρ, "t_stat": t, "se": se}} + {"r_squared": R², "n_obs": n}
        실패 시 None
    """
    y = y[~np.isnan(y)]
    max_lag = max(lags)
    min_obs = max_lag + 3  # 최소 관측치: max_lag + 3

    if len(y) < min_obs:
        return None

    # 종속변수: R_t (max_lag 이후부터)
    Y = y[max_lag:]
    n = len(Y)

    # 독립변수: [1, R_{t-lag1}, R_{t-lag2}, ...]
    X_cols = [np.ones(n)]  # intercept
    for lag in lags:
        X_cols.append(y[max_lag - lag: -lag] if lag < len(y) else y[max_lag - lag:])
    X = np.column_stack(X_cols)

    try:
        XtX = X.T @ X
        XtY = X.T @ Y
        beta = np.linalg.solve(XtX, XtY)  # [α, ρ₁, ρ₂, ...]

        # 잔차
        resid = Y - X @ beta
        sse = resid @ resid
        k = len(lags) + 1  # intercept + lag 개수
        dof = n - k

        if dof <= 0:
            return None

        s2 = sse / dof
        var_beta = s2 * np.linalg.inv(XtX)

        # R²
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        r_squared = 1 - sse / ss_tot if ss_tot > 0 else 0.0

        result = {"r_squared": r_squared, "n_obs": n}

        for i, lag in enumerate(lags):
            coef = beta[i + 1]  # i+1 because beta[0] = intercept
            se = np.sqrt(var_beta[i + 1, i + 1])
            t_stat = coef / se if se > 0 else np.nan
            result[lag] = {"coef": coef, "t_stat": t_stat, "se": se}

        return result

    except np.linalg.LinAlgError:
        return None


def rolling_ar(series: pd.Series, window: int, lags: list, min_periods: int = None):
    """
    시계열에 대해 롤링 AR(p) 추정

    Parameters:
        series: 팩터 주간수익률 시계열
        window: 롤링 윈도우 크기 (주 단위)
        lags: lag 리스트 (예: [1], [1,2], [2], [1,2,4])
        min_periods: 최소 관측치 수

    Returns:
        DataFrame with columns: lag별 coef/t_stat + r_squared, n_obs, signal
    """
    if min_periods is None:
        min_periods = max(max(lags) + 3, int(window * 0.7))

    dates = series.index
    results = []

    for i in range(len(series)):
        start_idx = max(0, i - window + 1)
        window_data = series.iloc[start_idx:i + 1].values

        valid_count = np.sum(~np.isnan(window_data))

        row = {"date": dates[i]}

        if valid_count >= min_periods:
            res = ols_ar(window_data, lags)
            if res is not None:
                row["r_squared"] = res["r_squared"]
                row["n_obs"] = res["n_obs"]
                for lag in lags:
                    row[f"coef_lag{lag}"] = res[lag]["coef"]
                    row[f"t_stat_lag{lag}"] = res[lag]["t_stat"]
            else:
                row["r_squared"] = np.nan
                row["n_obs"] = 0
                for lag in lags:
                    row[f"coef_lag{lag}"] = np.nan
                    row[f"t_stat_lag{lag}"] = np.nan
        else:
            row["r_squared"] = np.nan
            row["n_obs"] = 0
            for lag in lags:
                row[f"coef_lag{lag}"] = np.nan
                row[f"t_stat_lag{lag}"] = np.nan

        results.append(row)

    return pd.DataFrame(results).set_index("date")


# ──────────────────────────────────────────────
# 4. 전체 팩터에 대해 롤링 AR 실행
# ──────────────────────────────────────────────
def run_rolling_ar_all_factors(
    pivot: pd.DataFrame,
    window: int = 12,
    lags: list = None,
    t_threshold: float = 2.0,
    display_start: str = "20241201",
) -> dict:
    """
    Parameters:
        pivot: 날짜 × 팩터 주간수익률 DataFrame
        window: 롤링 윈도우 (주 단위)
        lags: lag 리스트 (기본 [1])
        t_threshold: t-stat 임계값
        display_start: 차트 표시 시작일

    Returns:
        dict[factor_name] -> DataFrame
    """
    if lags is None:
        lags = [1]

    existing = [f for f in FACTOR_ORDER if f in pivot.columns]
    ar_results = {}

    lag_str = ", ".join([f"{l}W" for l in lags])
    print(f"\n{'='*65}")
    print(f"AR({len(lags)}) Rolling Estimation  |  Lags: [{lag_str}]")
    print(f"  Window: {window}W | t-stat threshold: {t_threshold}")
    print(f"{'='*65}")

    for factor in existing:
        print(f"  ▶ {factor} 추정 중...", end=" ")
        res = rolling_ar(pivot[factor], window=window, lags=lags)

        # 모멘텀 신호: 모든 lag에서 coef > 0 AND t_stat > threshold
        signal_conditions = []
        for lag in lags:
            cond = (res[f"coef_lag{lag}"] > 0) & (res[f"t_stat_lag{lag}"] > t_threshold)
            signal_conditions.append(cond)

        # 하나라도 유의한 양의 자기상관이 있으면 signal ON
        res["signal_any"] = pd.concat(signal_conditions, axis=1).any(axis=1)
        # 모든 lag에서 유의해야 signal ON
        res["signal_all"] = pd.concat(signal_conditions, axis=1).all(axis=1)

        ar_results[factor] = res

        any_count = res["signal_any"].sum()
        all_count = res["signal_all"].sum()
        print(f"완료 (any={any_count}건, all={all_count}건)")

    return ar_results


# ──────────────────────────────────────────────
# 5. 시각화
# ──────────────────────────────────────────────
def plot_ar_results(
    ar_results: dict,
    pivot: pd.DataFrame,
    lags: list = None,
    t_threshold: float = 2.0,
    display_start: str = "20241201",
    window: int = 12,
    signal_mode: str = "any",
):
    if lags is None:
        lags = [1]

    existing = [f for f in FACTOR_ORDER if f in ar_results]
    signal_col = f"signal_{signal_mode}"
    lag_str = ", ".join([f"{l}W" for l in lags])

    n_panels = 2 + len(lags) + 1  # coef panels + t_stat panels + R² + heatmap
    # 구성: lag별 coef, lag별 t_stat, R², 히트맵
    n_panels = 1 + len(lags) + 1 + 1  # coef(합쳐서1개) + t_stat(lag별) + R² + heatmap

    fig, axes = plt.subplots(
        3 + 1, 1, figsize=(18, 5 * 3 + 3),
        gridspec_kw={"height_ratios": [2.5] + [2] * 2 + [1.2]},
    )

    fig.suptitle(
        f"Factor AR Autocorrelation Analysis  |  Lags=[{lag_str}]\n"
        f"$R_t = \\alpha + \\sum \\rho_i \\cdot R_{{t-i}} + \\epsilon$"
        f"  |  Window={window}W  |  t-threshold={t_threshold}",
        fontsize=15, fontweight="bold", y=0.995,
    )

    start_dt = pd.to_datetime(display_start)

    # ── Panel 1: Rolling ρ (각 lag별 계수, 팩터별) ──
    ax1 = axes[0]
    for factor in existing:
        df = ar_results[factor]
        df_disp = df[df.index >= start_dt]
        for lag in lags:
            col = f"coef_lag{lag}"
            linestyle = "-" if lag == lags[0] else "--" if lag == lags[-1] else "-."
            alpha = 1.0 if len(lags) == 1 else 0.8
            label = f"{factor}" if len(lags) == 1 else f"{factor} (lag{lag})"
            ax1.plot(df_disp.index, df_disp[col],
                     label=label, color=COLORS[factor],
                     linewidth=1.5, linestyle=linestyle, alpha=alpha)

    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax1.set_title(f"① Rolling AR Coefficient (ρ)  |  Lags=[{lag_str}]", fontsize=13, pad=8)
    ax1.set_ylabel("ρ (autocorrelation)")
    ax1.legend(loc="upper left", fontsize=8, ncol=min(6, len(existing) * len(lags)), framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.8, 0.8)

    # ── Panel 2: Rolling t-stat (각 lag별) ──
    ax2 = axes[1]
    for factor in existing:
        df = ar_results[factor]
        df_disp = df[df.index >= start_dt]
        for lag in lags:
            col = f"t_stat_lag{lag}"
            linestyle = "-" if lag == lags[0] else "--" if lag == lags[-1] else "-."
            label = f"{factor}" if len(lags) == 1 else f"{factor} (lag{lag})"
            ax2.plot(df_disp.index, df_disp[col],
                     label=label, color=COLORS[factor],
                     linewidth=1.5, linestyle=linestyle, alpha=0.8)

    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax2.axhline(t_threshold, color="red", linewidth=1.2, linestyle="--",
                alpha=0.7, label=f"threshold ({t_threshold})")
    ax2.axhline(-t_threshold, color="blue", linewidth=1.2, linestyle="--",
                alpha=0.7, label=f"threshold (-{t_threshold})")
    ax2.set_title(f"② Rolling t-statistic (threshold = ±{t_threshold})", fontsize=13, pad=8)
    ax2.set_ylabel("t-stat")
    ax2.legend(loc="upper left", fontsize=8, ncol=4, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Rolling R² ──
    ax3 = axes[2]
    for factor in existing:
        df = ar_results[factor]
        df_disp = df[df.index >= start_dt]
        ax3.plot(df_disp.index, df_disp["r_squared"],
                 label=factor, color=COLORS[factor], linewidth=1.5)

    ax3.set_title("③ Rolling R² (설명력)", fontsize=13, pad=8)
    ax3.set_ylabel("R²")
    ax3.legend(loc="upper left", fontsize=9, ncol=6, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.05, 0.5)

    # ── Panel 4: 신호 히트맵 ──
    ax4 = axes[3]
    signal_data = []
    for factor in existing:
        df = ar_results[factor]
        df_disp = df[df.index >= start_dt]
        signal_data.append(df_disp[signal_col].astype(int).values)

    dates = ar_results[existing[0]]
    dates = dates[dates.index >= start_dt].index

    heatmap = np.array(signal_data)
    cmap = ListedColormap(["#ffcccc", "#66bb6a"])
    ax4.imshow(heatmap, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    ax4.set_yticks(range(len(existing)))
    ax4.set_yticklabels(existing, fontsize=10)

    tick_step = max(1, len(dates) // 15)
    tick_positions = list(range(0, len(dates), tick_step))
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels([dates[i].strftime("%m-%d") for i in tick_positions],
                        rotation=45, fontsize=8)

    mode_label = "ANY lag" if signal_mode == "any" else "ALL lags"
    ax4.set_title(
        f"④ Momentum Signal ({mode_label}: ρ > 0 AND t > {t_threshold})",
        fontsize=13, pad=6,
    )

    # 최신 상태 라벨
    for i, factor in enumerate(existing):
        df = ar_results[factor]
        df_disp = df[df.index >= start_dt]
        if not df_disp.empty:
            latest = df_disp.iloc[-1]
            status = "✅ ON" if latest[signal_col] else "❌ OFF"
            color = "#2e7d32" if latest[signal_col] else "#c62828"
            ax4.text(len(dates) + 0.5, i, f" {status}", va="center",
                     fontsize=10, fontweight="bold", color=color)

    for ax in axes[:3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout(rect=[0, 0, 0.95, 0.97])
    plt.savefig("factor_ar_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ 차트 저장: factor_ar_analysis.png")


# ──────────────────────────────────────────────
# 6. 요약 출력
# ──────────────────────────────────────────────
def print_ar_summary(ar_results: dict, lags: list = None, t_threshold: float = 2.0, signal_mode: str = "any"):
    if lags is None:
        lags = [1]

    existing = [f for f in FACTOR_ORDER if f in ar_results]
    signal_col = f"signal_{signal_mode}"
    lag_str = ", ".join([f"{l}W" for l in lags])

    print("\n" + "=" * 90)
    print(f"📊 AR 자기상관 검증 결과 (최신 시점)  |  Lags=[{lag_str}]  |  t-threshold={t_threshold}")
    print("=" * 90)

    # 최신 날짜
    latest_date = None
    for df in ar_results.values():
        if not df.empty:
            latest_date = df.index[-1]
            break

    if latest_date:
        print(f"  기준일: {latest_date.strftime('%Y-%m-%d')}\n")

    # 헤더
    header = f"  {'팩터':<12}"
    for lag in lags:
        header += f"{'ρ_lag'+str(lag):>10} {'t_lag'+str(lag):>10}"
    header += f" {'R²':>8} {'n':>5}  {'판정':<20}"
    print(header)
    print("  " + "-" * (len(header) + 5))

    selected = []
    for factor in existing:
        df = ar_results[factor]
        latest = df.iloc[-1]

        row = f"  {factor:<12}"

        lag_signals = []
        for lag in lags:
            coef = latest[f"coef_lag{lag}"]
            t_stat = latest[f"t_stat_lag{lag}"]
            row += f"{coef:>10.4f} {t_stat:>10.3f}"
            lag_signals.append(coef > 0 and t_stat > t_threshold)

        r_sq = latest["r_squared"]
        n_obs = int(latest["n_obs"])
        signal = latest[signal_col]

        row += f" {r_sq:>8.4f} {n_obs:>5}"

        if signal:
            verdict = "✅ 모멘텀 유의"
            selected.append(factor)
        elif any(latest[f"coef_lag{l}"] > 0 for l in lags):
            verdict = "⚠️ 양이나 비유의"
        elif any(latest[f"coef_lag{l}"] < 0 and abs(latest[f"t_stat_lag{l}"]) > t_threshold for l in lags):
            verdict = "🔄 반전 신호"
        else:
            verdict = "❌ 자기상관 없음"

        row += f"  {verdict}"
        print(row)

    print("\n" + "-" * 90)
    mode_label = "하나라도" if signal_mode == "any" else "모든 lag에서"
    if selected:
        print(f"  🔥 모멘텀 유의 팩터: {', '.join(selected)}")
        print(f"     → {mode_label} ρ>0, t>{t_threshold} 충족")
    else:
        print(f"  📉 현재 기준을 충족하는 모멘텀 팩터가 없습니다.")

    # 전체 기간 신호 비율
    print(f"\n  [전체 기간 신호 ON 비율 ({signal_col})]")
    for factor in existing:
        df = ar_results[factor]
        on_ratio = df[signal_col].mean() * 100
        bar_len = int(on_ratio / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {factor:<12} {bar} {on_ratio:.1f}%")

    print("\n" + "=" * 90)

    return selected


# ──────────────────────────────────────────────
# 7. 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":

    # =============================================
    # ▼▼▼ 여기서 파라미터 조절 ▼▼▼
    # =============================================
    ROLLING_WINDOW = 10        # 롤링 윈도우 (주 단위)
    T_THRESHOLD = 2.0          # t-stat 임계값
    DISPLAY_START = "20231201"  # 차트 표시 시작일
    DATA_START = "20040201"     # 데이터 조회 시작일
    DATA_END = "20251219"       # 데이터 조회 종료일

    # ▼▼▼ LAG 설정 ▼▼▼
    # [1]       → AR(1): 1주 전만 (기본)
    # [2]       → 2주 전만
    # [1, 2]    → AR(2): 1주 전 + 2주 전
    # [1, 2, 4] → 1주 + 2주 + 4주 전
    # [4]       → 4주 전만
    LAGS = [1]

    # ▼▼▼ 신호 판정 모드 ▼▼▼
    # "any" → lag 중 하나라도 유의하면 ON
    # "all" → 모든 lag에서 유의해야 ON
    SIGNAL_MODE = "any"
    # =============================================

    lag_str = ", ".join([f"{l}W" for l in LAGS])
    print(f"📊 팩터 AR 자기상관 롤링 검증 시작...\n")
    print(f"  설정: window={ROLLING_WINDOW}W, lags=[{lag_str}], t_threshold={T_THRESHOLD}")
    print(f"  신호 모드: {SIGNAL_MODE}")
    print(f"  기간: {DATA_START} ~ {DATA_END}")
    print(f"  표시: {DISPLAY_START} ~\n")

    df = fetch_factor_data(DATA_START, DATA_END)

    if not df.empty:
        print(f"✅ 데이터 로드: {len(df)}건")

        pivot = (
            df.pivot_table(index="BaseDate", columns="FactorName",
                           values="LnRtn_L_S", aggfunc="first")
            .sort_index()
        )
        existing = [f for f in FACTOR_ORDER if f in pivot.columns]
        pivot = pivot[existing]

        # AR 롤링 추정
        ar_results = run_rolling_ar_all_factors(
            pivot,
            window=ROLLING_WINDOW,
            lags=LAGS,
            t_threshold=T_THRESHOLD,
            display_start=DISPLAY_START,
        )

        # 시각화
        plot_ar_results(
            ar_results, pivot,
            lags=LAGS,
            t_threshold=T_THRESHOLD,
            display_start=DISPLAY_START,
            window=ROLLING_WINDOW,
            signal_mode=SIGNAL_MODE,
        )

        # 요약 및 팩터 선별
        selected = print_ar_summary(
            ar_results,
            lags=LAGS,
            t_threshold=T_THRESHOLD,
            signal_mode=SIGNAL_MODE,
        )

    else:
        print("❌ 데이터를 가져오지 못했습니다.")

# python Momentum.py