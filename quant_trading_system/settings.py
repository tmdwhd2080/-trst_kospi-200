"""
settings.py — 퀀트 트레이딩 시스템 통합 설정

봇 전체에 관련된 설정만 여기서 관리
각 전략/스크래퍼 개별 설정은 해당 모듈 파일 상단에서 조절
"""

#  주문 설정
DRY_RUN = True          # True: 시뮬레이션 (주문 안 나감) / False: 실전 주문

#  API 설정 (Gemini / GPT 선택)

AI_PROVIDER = "gemini"                  # "gemini" 또는 "gpt"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"  # Google Gemini API 키
GEMINI_MODEL = "gemini-2.0-flash"       # Gemini 모델명
GPT_API_KEY = "YOUR_GPT_API_KEY"        # OpenAI GPT API 키
GPT_MODEL = "gpt-4o-mini"              # GPT 모델명
USE_AI_BRIEFING = True                  # AI 시장 브리핑 & 종가 예측 기능

#  3. 한국투자증권 API 키
KIS_APP_KEY = "PSbk6Xl0PUXbUqVljrmhFwpMg2jBY4DPC41C"
KIS_APP_SECRET = "t/pAxqo5un71gPnp2hR6bN580V6Iwc+NEKqHtlp21JuSykAxdAxqUofZNtMeMJqNsIcMWParg5cL4tte/FoZzxDygMB64K98eIhIum3HBEj3KL8BRWZWlagM4siYb0wdX2Gut7z5Mz5V7tzX+EZ747PKM1293NIjLn2XW+4T502Pqqbx1Oo="
KIS_ACCOUNT_NO = "71022030"     # 계좌번호 앞 8자리
KIS_ACCOUNT_CODE = "01"                 # 계좌 상품코드
KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"

#  장중 OHLCV 스크래핑 설정
STOCK_SCRAPPING_GROUPS = ["kospi"]      # 기본 추적 그룹 (stock_list.json 기준)
STOCK_SCRAPPING_SYMBOLS = []            # 비워두면 그룹 전체에서 선택
STOCK_SCRAPPING_LIMIT = 0               # 0이면 scrapers/stock_scrapping.py의 안전 상한 자동 적용
STOCK_SCRAPPING_BATCH_SIZE = 250
STOCK_SCRAPPING_POLL_INTERVAL_SEC = 300
STOCK_SCRAPPING_MIN_REQUEST_GAP_SEC = 1.0
STOCK_SCRAPPING_MAX_RETRIES = 5
STOCK_SCRAPPING_OUTPUT = "live_ohlcv_5m.csv"
STOCK_SCRAPPING_EXIT_AFTER_SESSION = True

#  전략 및 스크래퍼 활성화 여부 설정
ENABLE_TASKS = {
    "krx_disclosure":     False,   # KRX KIND 공시 크롤링
    "macro_indicators":   True,    # 매크로 지표 수집 (VIX, 환율, 심리지수)
    "ai_briefing":        True,    # AI 시장 브리핑 & 종가 예측 (아침)
    "factor_momentum":    True,    # 팩터/모멘텀 전략
    "hrl_allocation":     False,   # HRL 자산 배분 전략
    "ai_closing_review":  True,    # AI 장 마감 리뷰 (예측 vs 실제 비교)
    "stock_scrapping":    True,    # 장중 주가 OHLCV 수집
    # __INTEGRATE_MARKER_ENABLE_TASKS__
}

#  5. 기본 스케줄 (하드코딩)
#     ※ ENABLE_TASKS에서 비활성화된 작업은 자동 제외됩니다
DEFAULT_SCHEDULE = [
    {"time": "08:50", "task": "macro_indicators",  "description": "장 시작 전 매크로 지표 수집"},
    {"time": "08:55", "task": "ai_briefing",        "description": "AI 시장 브리핑 & 종가 예측"},
    {"time": "09:00", "task": "stock_scrapping",    "description": "장중 주가 OHLCV 수집", "force_kill_time": "15:30"},
    {"time": "09:05", "task": "krx_disclosure",     "description": "장 시작 후 공시 수집"},
    {"time": "09:10", "task": "factor_momentum",    "description": "팩터/모멘텀 전략 실행", "force_kill_time": "09:40"},
    {"time": "09:15", "task": "hrl_allocation",     "description": "HRL 자산 배분 전략 실행"},
    {"time": "12:00", "task": "krx_disclosure",     "description": "점심 공시 수집"},
    {"time": "12:05", "task": "macro_indicators",   "description": "점심 매크로 지표 갱신"},
    {"time": "15:20", "task": "krx_disclosure",     "description": "장 마감 전 공시 수집"},
    {"time": "15:35", "task": "macro_indicators",   "description": "장 마감 후 매크로 지표 최종 수집"},
    {"time": "15:45", "task": "ai_closing_review",  "description": "AI 장 마감 리뷰 (예측 vs 실제 비교)"},
    # _{"time": "09:10", "task": "factor_momentum",    "description": "팩터/모멘텀 전략 실행", "force_kill_time": "09:40"} -< 이런식으로 강제 종료 시간 설정 가능
    # __INTEGRATE_MARKER_DEFAULT_SCHEDULE__
]

#  6. 스케줄러 설정
SCHEDULE_CHECK_INTERVAL_SEC = 60    # 60초마다 스케줄 확인 -> 늘리거나 줄여도 됨
ERROR_RETRY_DELAY_SEC = 60          # 에러 발생 시 재시도 대기 (초)


#  7. KIS API 동작 설정

TOKEN_VALIDITY_SEC = 82800          # 토큰 유효 시간 (초, 23시간)
API_TIMEOUT_SEC = 10                # API 요청 타임아웃 (초)


#  8. 서버 배포 설정 (run.py --deploy 에서 사용) -> 자동으로 서버 열림

SERVER = {
    "host": "115.68.232.108",             # 서버 IP 주소 (run.py 실행 시 입력받음)
    "user": "root",         # SSH 접속 유저
    "ssh_key": "",          # SSH 키 경로 (비워두면 기본 키 사용, 예: C:/Users/유저/.ssh/id_rsa)
    "remote_dir": "/root/quant_trading_system",     # 서버 내 프로젝트 경로
    "tmux_session": "trade_bot",                    # tmux 세션 이름
}
