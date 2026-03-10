"""
settings.py — 퀀트 트레이딩 시스템 통합 설정
═══════════════════════════════════════════════
모든 하드코딩 수치를 여기서 관리합니다.
이 파일만 수정하면 전체 시스템에 반영됩니다.
"""

# ═══════════════════════════════════════════════════
#  1. 실행 모드
# ═══════════════════════════════════════════════════
DRY_RUN = True          # True: 시뮬레이션 (주문 안 나감) / False: 실전 주문

# ═══════════════════════════════════════════════════
#  2. Gemini AI 설정
# ═══════════════════════════════════════════════════
GEMINI_MODEL = "gemini-2.0-flash"       # 사용할 Gemini 모델
USE_GEMINI_SCHEDULER = True             # True: Gemini가 스케줄 생성 / False: 기본 스케줄 사용
USE_GEMINI_ADVISOR = True               # True: 작업 결과를 Gemini에게 전달하여 후속 판단

# ═══════════════════════════════════════════════════
#  3. 작업 활성화 ON/OFF
# ═══════════════════════════════════════════════════
ENABLE_TASKS = {
    "krx_disclosure":   True,   # KRX KIND 공시 크롤링
    "macro_indicators": True,   # 매크로 지표 수집 (VIX, 환율, 심리지수)
    "factor_momentum":  True,   # 팩터/모멘텀 전략
    "hrl_allocation":   True,   # HRL 자산 배분 전략
}

# ═══════════════════════════════════════════════════
#  4. 기본 스케줄 (Gemini 미사용 시 또는 폴백)
#     ※ ENABLE_TASKS에서 비활성화된 작업은 자동 제외됩니다
# ═══════════════════════════════════════════════════
DEFAULT_SCHEDULE = [
    {"time": "08:50", "task": "macro_indicators", "description": "장 시작 전 매크로 지표 수집"},
    {"time": "09:05", "task": "krx_disclosure",   "description": "장 시작 후 공시 수집"},
    {"time": "09:10", "task": "factor_momentum",  "description": "팩터/모멘텀 전략 실행"},
    {"time": "09:15", "task": "hrl_allocation",   "description": "HRL 자산 배분 전략 실행"},
    {"time": "12:00", "task": "krx_disclosure",   "description": "점심 공시 수집"},
    {"time": "12:05", "task": "macro_indicators", "description": "점심 매크로 지표 갱신"},
    {"time": "15:20", "task": "krx_disclosure",   "description": "장 마감 전 공시 수집"},
    {"time": "15:35", "task": "macro_indicators", "description": "장 마감 후 매크로 지표 최종 수집"},
]

# ═══════════════════════════════════════════════════
#  5. 스케줄러 설정
# ═══════════════════════════════════════════════════
SCHEDULE_CHECK_INTERVAL_SEC = 30    # 메인 루프 스케줄 체크 주기 (초)
ERROR_RETRY_DELAY_SEC = 60          # 에러 발생 시 재시도 대기 (초)

# ═══════════════════════════════════════════════════
#  6. KIS API 설정
# ═══════════════════════════════════════════════════
TOKEN_VALIDITY_SEC = 82800          # 토큰 유효 시간 (초, 23시간)
API_TIMEOUT_SEC = 10                # API 요청 타임아웃 (초)

# ═══════════════════════════════════════════════════
#  7. KRX 공시 크롤러 설정
# ═══════════════════════════════════════════════════
KRX_CRAWL_PAGE_SIZE = 100           # 한 번에 가져올 공시 건수
KRX_ALERT_KEYWORDS = [
    "유상증자", "무상증자", "합병", "분할", "자사주",
    "배당", "대규모내부거래", "공개매수", "상장폐지",
    "액면분할", "전환사채", "신주인수권",
]

# ═══════════════════════════════════════════════════
#  8. 팩터/모멘텀 전략 설정
# ═══════════════════════════════════════════════════
FACTOR_UNIVERSE = [
    "005930",   # 삼성전자
    "000660",   # SK하이닉스
    "035420",   # NAVER
    "005380",   # 현대차
    "051910",   # LG화학
]
FACTOR_MAX_STOCKS = 3               # 최대 편입 종목 수
FACTOR_ALLOCATION_PCT = 0.30        # 종목당 최대 투자 비중 (0.0 ~ 1.0)
FACTOR_ORDER_QTY = 1                # 종목당 기본 주문 수량

# ═══════════════════════════════════════════════════
#  9. HRL 자산 배분 전략 설정
# ═══════════════════════════════════════════════════
HRL_ASSET_UNIVERSE = {
    "equity":   ["005930", "000660", "035420"],     # 주식
    "bond_etf": ["148070"],                          # 채권 ETF
    "gold_etf": ["411060"],                          # 금 ETF
}
HRL_REGIME_RULES = {
    "Extreme Fear": {"equity": 0.2, "bond_etf": 0.5, "gold_etf": 0.3},
    "Fear":         {"equity": 0.3, "bond_etf": 0.4, "gold_etf": 0.3},
    "Neutral":      {"equity": 0.5, "bond_etf": 0.3, "gold_etf": 0.2},
    "Greed":        {"equity": 0.7, "bond_etf": 0.2, "gold_etf": 0.1},
    "Extreme Greed":{"equity": 0.8, "bond_etf": 0.1, "gold_etf": 0.1},
}
HRL_ORDER_QTY = 1                   # 종목당 기본 주문 수량

# ═══════════════════════════════════════════════════
#  10. 매크로 지표 수집 설정
# ═══════════════════════════════════════════════════
VIX_THRESHOLDS = {
    "extreme_greed": 12,    # 이하 → Extreme Greed
    "greed": 18,            # 이하 → Greed
    "neutral": 25,          # 이하 → Neutral
    "fear": 30,             # 이하 → Fear
    # 초과 → Extreme Fear
}

# ═══════════════════════════════════════════════════
#  11. 서버 배포 설정 (run.py에서 사용)
# ═══════════════════════════════════════════════════
SERVER = {
    "host": "",             # 서버 IP 주소 (run.py 실행 시 입력받음)
    "user": "root",         # SSH 접속 유저
    "remote_dir": "/root/quant_trading_system",     # 서버 내 프로젝트 경로
    "tmux_session": "trade_bot",                    # tmux 세션 이름
}
