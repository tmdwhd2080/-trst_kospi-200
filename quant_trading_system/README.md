# Quant Trading System

퀀트 트레이딩 자동화 시스템.  
AI(Gemini/GPT)가 스케줄을 관리하고, 크롤러와 전략을 자동 실행합니다.

---

## 프로젝트 구조

```
quant_trading_system/
├── settings.py                       #  봇 전체 설정 + API 키 (하드코딩)
├── requirements.txt                  # Python 라이브러리 목록
├── main_scheduler.py                 #  메인 스케줄러 (이것만 실행)
├── run.py                            # 로컬 실행 / 서버 배포 도구
├── utils/
│   ├── __init__.py
│   └── kis_api.py                    # 한국투자증권 API 공통 함수
├── scrapers/
│   ├── __init__.py
│   ├── krx_kind_crawler.py           # [예시] KRX 공시 모니터링 크롤러
│   └── macro_indicator_scraper.py    # [예시] VIX, 환율, 공포탐욕지수 수집
└── strategies/
    ├── __init__.py
    ├── factor_momentum_allocation.py # [예시] 팩터/모멘텀 전략
    └── hrl_asset_allocation.py       # [예시] HRL 자산 배분 전략
```


## 빠른 시작

### 1. 사전 준비

| 항목 | 설명 |
|---|---|
| **Python** | 3.10 이상 |
| **한국투자증권 API** | [KIS Developers](https://apiportal.koreainvestment.com/) 에서 APP KEY/SECRET 발급 |
| **AI API (택1)** | [Google AI Studio](https://aistudio.google.com/) 또는 [OpenAI](https://platform.openai.com/) 에서 API 키 발급 |
| **서버 (선택)** | Oracle Linux, Ubuntu 등 24시간 가동 서버 |

### 2. settings.py 설정

모든 API 키와 동작 설정은 `settings.py`에 하드코딩되어 있습니다.  
**별도의 `.env` 파일 없이** `settings.py`만 수정하면 됩니다:

```python
# AI 제공자 선택 ("gemini" 또는 "gpt")
AI_PROVIDER = "gemini"
GEMINI_API_KEY = "발급받은_GEMINI_KEY"
GPT_API_KEY = "발급받은_GPT_KEY"

# 한국투자증권 API
KIS_APP_KEY = "발급받은_APP_KEY"
KIS_APP_SECRET = "발급받은_APP_SECRET"
KIS_ACCOUNT_NO = "계좌번호앞8자리"
KIS_ACCOUNT_CODE = "01"

# 시뮬레이션 모드 (True: 주문 안 나감 / False: 실전 주문)
DRY_RUN = True

# 작업 활성화 ON/OFF
ENABLE_TASKS = {
    "krx_disclosure":   True,
    "macro_indicators": True,
    "factor_momentum":  True,
    "hrl_allocation":   True,
}

# AI 사용 여부
USE_AI_SCHEDULER = True     # AI가 스케줄 생성
USE_AI_ADVISOR = True       # 작업 결과 분석 및 후속 조치 판단
```

> ⚠ **중요**: 실전 주문을 실행하려면 `DRY_RUN = False`로 변경하세요.  
> 반드시 시뮬레이션 모드에서 충분히 테스트한 후 실전 모드로 전환하세요.

### 3. 전략별 개별 설정

각 전략/스크래퍼의 파라미터는 **해당 모듈 파일 상단**에서 직접 조절합니다:

| 파일 | 설정 항목 |
|---|---|
| `strategies/factor_momentum_allocation.py` | UNIVERSE, MAX_STOCKS, ALLOCATION_PCT, ORDER_QTY |
| `strategies/hrl_asset_allocation.py` | ASSET_UNIVERSE, REGIME_RULES, ORDER_QTY |
| `scrapers/krx_kind_crawler.py` | CRAWL_PAGE_SIZE, ALERT_KEYWORDS |
| `scrapers/macro_indicator_scraper.py` | VIX_THRESHOLDS |

---

## 사용 방법

### 로컬에서 실행 (기본)

```bash
# 1. 라이브러리 설치
pip install -r requirements.txt

# 2. 실행 (run.py만 실행하면 바로 봇 시작)
python run.py
```

실행하면 아래와 같은 배너가 출력됩니다:

```
╔══════════════════════════════════════════════════╗
║      QUANT TRADING SYSTEM — CONTROL TOWER       ║
╠══════════════════════════════════════════════════╣

  모드: 🟢 시뮬레이션 (DRY RUN)
  활성 작업: krx_disclosure, macro_indicators, factor_momentum, hrl_allocation
  Gemini 스케줄러: ON
  Gemini 어드바이저: ON
```

### 서버 배포

`run.py --deploy`를 실행하면 서버 접속부터 봇 실행까지 자동으로 처리됩니다:

```bash
python run.py --deploy
```

실행 과정:
1. 서버 IP / 비밀번호 입력
2. 서버에 Python, tmux 등 패키지 자동 설치
3. 프로젝트 파일 전체 업로드 (scp)
4. Python 라이브러리 설치 (pip)
5. tmux 세션에서 `main_scheduler.py` 자동 실행

```
서버 IP 주소: xxx.xxx.xxx.xxx
root@xxx.xxx.xxx.xxx 비밀번호: ********

[1/5] 서버 패키지 설치 중...
[2/5] 원격 디렉토리 생성...
[3/5] 프로젝트 파일 업로드 중...
[4/5] Python 라이브러리 설치 중...
[5/5] tmux 세션에서 봇 실행 중...

🚀 배포 완료!
```

### 서버 관리 명령어

```bash
# 봇 상태 확인
python run.py --status

# 최근 로그 확인
python run.py --logs

# 봇 중지
python run.py --stop
```

### 수동 서버 관리 (SSH 직접 접속)

```bash
# 서버 접속
ssh root@서버IP

# 봇 터미널로 진입
tmux attach -t trade_bot

# 봇 중지: Ctrl+C
# 터미널 분리 (봇은 계속 돌아감): Ctrl+B → D

# 서버 나가기 (반드시 exit 입력)
exit
```

---

## 동작 원리

### 전체 플로우

```
main_scheduler.py 실행
      │
      ├─ [시작] KIS 토큰 사전 발급
      │
      ├─ [AI] 오늘 날짜/요일/작업 목록 전달 → 스케줄 JSON 생성
      │           (실패 시 settings.py의 DEFAULT_SCHEDULE 사용)
      │
      ├─ [스케줄 등록] 비활성화 작업 자동 제외
      │
      └─ [메인 루프] 30초마다 스케줄 체크
              │
              ├─ 시간 도달 → 해당 모듈 import → run() 실행
              │       │
              │       └─ [AI 어드바이저] 결과 분석 → 후속 조치 판단
              │               - ACTION: NONE (조치 없음)
              │               - ACTION: RUN <task_id> (추가 작업 실행)
              │               - ACTION: ALERT <이유> (경고 로그)
              │
              └─ 자정 → 날짜 변경 감지 → 스케줄 재생성
```

### 기본 스케줄

| 시간 | 작업 | 설명 |
|---|---|---|
| 08:50 | `macro_indicators` | 장 시작 전 매크로 지표 수집 |
| 09:05 | `krx_disclosure` | 장 시작 후 공시 수집 |
| 09:10 | `factor_momentum` | 팩터/모멘텀 전략 실행 |
| 09:15 | `hrl_allocation` | HRL 자산 배분 전략 실행 |
| 12:00 | `krx_disclosure` | 점심 공시 수집 |
| 12:05 | `macro_indicators` | 점심 매크로 지표 갱신 |
| 15:20 | `krx_disclosure` | 장 마감 전 공시 수집 |
| 15:35 | `macro_indicators` | 장 마감 후 최종 수집 |

> Gemini/GPT가 활성화되어 있으면 주말/공휴일은 자동으로 빈 스케줄 반환 (작업 안 함)

---

## ⭐ 전략 필수 포함 요소 및 형식

새로운 전략을 추가할 때 반드시 아래 형식을 따라야 합니다.

### 1. 파일 위치
`strategies/` 폴더 아래에 `.py` 파일 생성

### 2. 파일 구조 (필수)

```python
"""
[전략명] 간략 설명
────────────────────
- 전략 개요 설명
⚠ 이 파일은 구조 예시입니다.
"""

# ═══════════════════════════════════════════════════
#  ⚙ 전략 개별 설정 — 이 영역의 값만 수정하세요
# ═══════════════════════════════════════════════════
MY_PARAM_1 = "value"         # 전략 고유 파라미터
MY_PARAM_2 = 100             # 전략 고유 파라미터
# ═══════════════════════════════════════════════════

import logging
from utils.kis_api import get_current_price, buy_limit_order

logger = logging.getLogger(__name__)


def run(macro_data: dict | None = None, dry_run: bool = True) -> dict:
    """
    전략 실행 진입점 (필수 함수).
    Args:
        macro_data: 매크로 지표 수집기 결과 (선택)
        dry_run: True면 주문 실행 안 함 (시뮬레이션만)
    Returns:
        dict: 전략 실행 결과 (로그/AI 어드바이저에 전달됨)
    """
    logger.info("=== 내 전략 실행 ===")
    # ... 전략 로직 구현 ...
    return {"result": "success"}
```

### 3. 필수 요소 체크리스트

| 요소 | 설명 | 필수 여부 |
|---|---|---|
| **독스트링** | 전략명, 개요, 주의사항 기재 | ✅ 필수 |
| **전략 개별 설정 영역** | `# ⚙ 전략 개별 설정` 구분선 안에 파라미터 정의 | ✅ 필수 |
| **`import logging`** | 로깅 설정 | ✅ 필수 |
| **`from utils.kis_api import ...`** | 주문/조회에 필요한 함수 import | 선택 |
| **`logger = logging.getLogger(__name__)`** | 로거 초기화 | ✅ 필수 |
| **`def run(macro_data, dry_run) -> dict`** | 스케줄러가 호출하는 진입점 함수 | ✅ 필수 |
| **`macro_data` 파라미터** | 매크로 지표 (VIX, 환율, 심리지수) | ✅ 필수 |
| **`dry_run` 파라미터** | True면 시뮬레이션, False면 실전 주문 | ✅ 필수 |
| **반환값 `dict`** | AI 어드바이저에 전달될 결과 | ✅ 필수 |
| **`settings.py` import 금지** | 전략 파라미터는 파일 상단에 직접 정의 | 귌장 |

### 4. 등록 방법

새 전략 파일을 만든 후, **2개 파일**만 수정하면 됩니다:

**`main_scheduler.py`** — TASK_REGISTRY에 전략 등록:
```python
TASK_REGISTRY = {
    ...
    "my_strategy": {
        "module": "strategies.my_strategy",
        "description": "내 전략 설명",
        "type": "strategy",
    },
}
```

**`settings.py`** — ENABLE_TASKS에 작업 추가 + DEFAULT_SCHEDULE에 시간 설정:
```python
ENABLE_TASKS = {
    ...
    "my_strategy": True,
}

DEFAULT_SCHEDULE = [
    ...
    {"time": "09:20", "task": "my_strategy", "description": "내 전략 실행"},
]
```

---

## settings.py 전체 설정 가이드

### 실행 모드

| 설정 | 기본값 | 설명 |
|---|---|---|
| `DRY_RUN` | `True` | `True`: 시뮬레이션 / `False`: 실전 주문 |

### AI 설정

| 설정 | 기본값 | 설명 |
|---|---|---|
| `AI_PROVIDER` | `"gemini"` | `"gemini"` 또는 `"gpt"` 선택 |
| `GEMINI_API_KEY` | `"YOUR_..."` | Google Gemini API 키 |
| `GEMINI_MODEL` | `"gemini-2.0-flash"` | Gemini 모델명 |
| `GPT_API_KEY` | `"YOUR_..."` | OpenAI GPT API 키 |
| `GPT_MODEL` | `"gpt-4o-mini"` | GPT 모델명 |
| `USE_AI_SCHEDULER` | `True` | AI 스케줄 생성 |
| `USE_AI_ADVISOR` | `True` | 작업 결과 AI 분석 |

### 한국투자증권 API

| 설정 | 설명 |
|---|---|
| `KIS_APP_KEY` | APP KEY |
| `KIS_APP_SECRET` | APP SECRET |
| `KIS_ACCOUNT_NO` | 계좌번호 앞 8자리 |
| `KIS_ACCOUNT_CODE` | 계좌 상품코드 (기본 `"01"`) |
| `KIS_BASE_URL` | API 서버 URL |

### 작업 ON/OFF

| 설정 | 기본값 | 설명 |
|---|---|---|
| `ENABLE_TASKS["krx_disclosure"]` | `True` | KRX 공시 크롤링 |
| `ENABLE_TASKS["macro_indicators"]` | `True` | 매크로 지표 수집 |
| `ENABLE_TASKS["factor_momentum"]` | `True` | 팩터/모멘텀 전략 |
| `ENABLE_TASKS["hrl_allocation"]` | `True` | HRL 자산 배분 전략 |

### 서버 배포

| 설정 | 기본값 | 설명 |
|---|---|---|
| `SERVER["host"]` | `""` | 서버 IP (비워두면 실행 시 입력) |
| `SERVER["user"]` | `"root"` | SSH 유저명 |
| `SERVER["remote_dir"]` | `"/root/quant_trading_system"` | 서버 내 프로젝트 경로 |
| `SERVER["tmux_session"]` | `"trade_bot"` | tmux 세션 이름 |

---

## 자주 하는 실수 / FAQ

### Q: 비밀번호를 2번 틀려서 IP가 차단됐어요
서버에 fail2ban이 설정되어 있을 수 있습니다. 30분 후 다시 시도하세요.

### Q: 실전 주문으로 전환하려면?
`settings.py`에서 `DRY_RUN = False`로 변경하세요.

### Q: 특정 전략만 끄고 싶어요
`settings.py`에서 해당 작업을 `False`로 변경:
```python
ENABLE_TASKS = {
    "krx_disclosure":   True,
    "macro_indicators": True,
    "factor_momentum":  False,  # ← 비활성화
    "hrl_allocation":   True,
}
```

### Q: Gemini/GPT 없이 사용할 수 있나요?
네. `settings.py`에서:
```python
USE_AI_SCHEDULER = False
USE_AI_ADVISOR = False
```
이렇게 설정하면 `DEFAULT_SCHEDULE`에 따라 동작합니다.

### Q: 새 전략을 추가하려면?
위의 **⭐ 전략 필수 포함 요소 및 형식** 섹션을 참고하세요.

### Q: 서버에서 봇이 돌아가는지 어떻게 확인하나요?
```bash
python run.py --status   # 로컬에서
# 또는 서버에 접속해서
tmux list-sessions
```

### Q: 로그는 어디에 저장되나요?
서버의 프로젝트 폴더에 `scheduler.log` 파일로 저장됩니다.
```bash
python run.py --logs     # 최근 50줄 확인
```

---

## 주의사항

- API 키가 포함된 `settings.py`를 **절대 공개 저장소에 커밋하지 마세요** (`.gitignore`에 등록 권장)
- 실전 주문 전에 반드시 `DRY_RUN = True` 상태에서 충분히 테스트하세요
- 서버 접속을 종료할 때는 반드시 `exit` 명령어를 입력하세요
