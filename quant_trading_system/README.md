# Quant Trading System

KOSPI 200 기반 퀀트 트레이딩 자동화 시스템.  
Gemini AI가 스케줄을 관리하고, 크롤러와 전략을 자동 실행합니다.

---

## 프로젝트 구조

```
quant_trading_system/
├── .env                              # API 키 보관 (Git 업로드 제외)
├── .gitignore                        # Git 제외 목록
├── settings.py                       # ⚙ 모든 설정값 통합 관리
├── requirements.txt                  # Python 라이브러리 목록
├── main_scheduler.py                 # ⏰ 메인 스케줄러 (이것만 실행)
├── run.py                            # 🚀 원클릭 서버 배포 도구
├── utils/
│   ├── __init__.py
│   └── kis_api.py                    # 한국투자증권 API 공통 함수
├── scrapers/
│   ├── __init__.py
│   ├── krx_kind_crawler.py           # KRX 공시 모니터링 크롤러
│   └── macro_indicator_scraper.py    # VIX, 환율, 공포탐욕지수 수집
└── strategies/
    ├── __init__.py
    ├── factor_momentum_allocation.py # [예시] 팩터/모멘텀 전략
    └── hrl_asset_allocation.py       # [예시] HRL 자산 배분 전략
```

---

## 빠른 시작

### 1. 사전 준비

| 항목 | 설명 |
|---|---|
| **Python** | 3.10 이상 |
| **한국투자증권 API** | [KIS Developers](https://apiportal.koreainvestment.com/) 에서 APP KEY/SECRET 발급 |
| **Google Gemini API** | [Google AI Studio](https://aistudio.google.com/) 에서 API 키 발급 |
| **서버 (선택)** | Oracle Linux, Ubuntu 등 24시간 가동 서버 |

### 2. API 키 설정

`.env` 파일을 열어 본인의 API 키를 입력합니다:

```env
# 한국투자증권 API
KIS_APP_KEY=발급받은_APP_KEY
KIS_APP_SECRET=발급받은_APP_SECRET
KIS_ACCOUNT_NO=계좌번호앞8자리
KIS_ACCOUNT_CODE=01

# Google Gemini API
GEMINI_API_KEY=발급받은_GEMINI_API_KEY
```

### 3. settings.py 설정

모든 동작 설정은 `settings.py`에서 관리합니다:

```python
# 시뮬레이션 모드 (True: 주문 안 나감 / False: 실전 주문)
DRY_RUN = True

# 작업 활성화 ON/OFF
ENABLE_TASKS = {
    "krx_disclosure":   True,   # KRX 공시 크롤링
    "macro_indicators": True,   # 매크로 지표 수집
    "factor_momentum":  True,   # 팩터/모멘텀 전략
    "hrl_allocation":   True,   # HRL 자산 배분 전략
}

# Gemini AI 사용 여부
USE_GEMINI_SCHEDULER = True     # Gemini가 스케줄 생성
USE_GEMINI_ADVISOR = True       # 작업 결과 분석 및 후속 조치 판단
```

> ⚠ **중요**: 실전 주문을 실행하려면 `DRY_RUN = False`로 변경하세요.  
> 반드시 시뮬레이션 모드에서 충분히 테스트한 후 실전 모드로 전환하세요.

---

## 사용 방법

### 로컬에서 실행

```bash
# 1. 라이브러리 설치
pip install -r requirements.txt

# 2. 실행
python main_scheduler.py
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

### 서버 배포 (원클릭)

`run.py`를 실행하면 서버 접속부터 봇 실행까지 자동으로 처리됩니다:

```bash
python run.py
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
      ├─ [Gemini] 오늘 날짜/요일/작업 목록 전달 → 스케줄 JSON 생성
      │           (실패 시 settings.py의 DEFAULT_SCHEDULE 사용)
      │
      ├─ [스케줄 등록] 비활성화 작업 자동 제외
      │
      └─ [메인 루프] 30초마다 스케줄 체크
              │
              ├─ 시간 도달 → 해당 모듈 import → run() 실행
              │       │
              │       └─ [Gemini 어드바이저] 결과 분석 → 후속 조치 판단
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

> Gemini가 활성화되어 있으면 주말/공휴일은 자동으로 빈 스케줄 반환 (작업 안 함)

---

## settings.py 전체 설정 가이드

### 실행 모드

| 설정 | 기본값 | 설명 |
|---|---|---|
| `DRY_RUN` | `True` | `True`: 시뮬레이션 / `False`: 실전 주문 |

### Gemini AI

| 설정 | 기본값 | 설명 |
|---|---|---|
| `GEMINI_MODEL` | `"gemini-2.0-flash"` | 사용할 Gemini 모델 |
| `USE_GEMINI_SCHEDULER` | `True` | AI 스케줄 생성 |
| `USE_GEMINI_ADVISOR` | `True` | 작업 결과 AI 분석 |

### 작업 ON/OFF

| 설정 | 기본값 | 설명 |
|---|---|---|
| `ENABLE_TASKS["krx_disclosure"]` | `True` | KRX 공시 크롤링 |
| `ENABLE_TASKS["macro_indicators"]` | `True` | 매크로 지표 수집 |
| `ENABLE_TASKS["factor_momentum"]` | `True` | 팩터/모멘텀 전략 |
| `ENABLE_TASKS["hrl_allocation"]` | `True` | HRL 자산 배분 전략 |

### 전략 파라미터

| 설정 | 기본값 | 설명 |
|---|---|---|
| `FACTOR_UNIVERSE` | 5종목 | 팩터 전략 종목 유니버스 |
| `FACTOR_MAX_STOCKS` | `3` | 최대 편입 종목 수 |
| `FACTOR_ALLOCATION_PCT` | `0.30` | 종목당 최대 투자 비중 |
| `FACTOR_ORDER_QTY` | `1` | 종목당 기본 주문 수량 |
| `HRL_ASSET_UNIVERSE` | 주식/채권/금 | HRL 전략 자산 유니버스 |
| `HRL_ORDER_QTY` | `1` | 종목당 기본 주문 수량 |

### 매크로 지표

| 설정 | 기본값 | 설명 |
|---|---|---|
| `VIX_THRESHOLDS` | 12/18/25/30 | VIX 기반 공포탐욕지수 기준값 |

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
`settings.py`에서 `DRY_RUN = False`로 변경 후 재배포하세요.  
```bash
python run.py
```

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

### Q: Gemini 없이 사용할 수 있나요?
네. `settings.py`에서:
```python
USE_GEMINI_SCHEDULER = False
USE_GEMINI_ADVISOR = False
```
이렇게 설정하면 `DEFAULT_SCHEDULE`에 따라 동작합니다.

### Q: 새 전략을 추가하려면?
1. `strategies/` 폴더에 새 파일 생성 (예: `my_strategy.py`)
2. `run(macro_data=None, dry_run=True)` 함수 구현
3. `main_scheduler.py`의 `TASK_REGISTRY`에 등록:
   ```python
   "my_strategy": {
       "module": "strategies.my_strategy",
       "description": "내 전략 설명",
       "type": "strategy",
   },
   ```
4. `settings.py`의 `ENABLE_TASKS`에 추가:
   ```python
   "my_strategy": True,
   ```
5. `DEFAULT_SCHEDULE`에 실행 시간 추가

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

- `.env` 파일에는 API 키가 포함되어 있으므로 **절대 Git에 커밋하지 마세요** (`.gitignore`에 이미 등록됨)
- 서버 IP와 비밀번호는 코드에 하드코딩하지 마세요
- 실전 주문 전에 반드시 `DRY_RUN = True` 상태에서 충분히 테스트하세요
- 서버 접속을 종료할 때는 반드시 `exit` 명령어를 입력하세요
