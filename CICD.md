# CI/CD Notes

- 2026-03-12

### 실질 변경사항

1. **`quant_trading_system/stock_scrapping.py` 신규 추가**
   - KIS API 기반 장중 OHLCV 수집기 추가
   - 종목 리스트(`stock_list.json`)를 읽어 KRX/US 종목을 세션 단위로 폴링(실시간 clock으로 개장 여부 확인)
   - 재시도/백오프, 세션 종료 후 토큰 폐기, CSV 적재 로직 포함
   - KIS API 한계로 인해 추후 종목 리스트 기반 필터링 추가 필요(현재 단순하게 JSON 상위부터 읽음)

2. **`quant_trading_system/scrapers/stock_scrapping.py` 신규 추가**
   - 메인 스케줄러가 호출할 수 있는 래퍼 엔트리포인트 추가
   - `settings.py`의 `STOCK_SCRAPPING_*` 설정을 읽어 실제 수집기에 전달
   - 봇이 장중에 재시작되어도 즉시 시작할 수 있도록 `should_start_now()` 제공

3. **`quant_trading_system/stock_list.json` 신규 추가**
   - `kospi`, `sp500` 그룹 기반 종목 메타데이터 추가
   - `stock_scrapping` 작업의 기본 심볼 유니버스로 사용

4. **`quant_trading_system/main_scheduler.py` 수정**
   - `stock_scrapping` 작업을 `TASK_REGISTRY`에 등록
   - 해당 작업을 백그라운드 스레드로 실행하도록 지원
   - 스케줄 항목의 `force_kill_time`을 로그/마지막 작업 시각 계산에 반영
   - 스케줄러가 장중에 켜진 경우, 활성 구간이면 `stock_scrapping`을 즉시 시작
   - 사전 발급한 KIS 토큰을 백그라운드 작업으로 전달

5. **`quant_trading_system/settings.py` 수정**
   - `STOCK_SCRAPPING_GROUPS`, `STOCK_SCRAPPING_LIMIT`, `STOCK_SCRAPPING_OUTPUT` 등 장중 수집 전용 설정 추가
   - `ENABLE_TASKS`에 `stock_scrapping` 추가
   - `DEFAULT_SCHEDULE`에 `09:00 ~ 15:30` 장중 수집 스케줄 추가

6. **`quant_trading_system/run.py` 수정**
   - 서버 배포 시 `stock_scrapping.py`, `stock_list.json`도 함께 업로드하도록 변경

7. **`quant_trading_system\requirements.txt` 수정**
   - 수집기 추가에 따른 tzdata, schedule 추가

8. **`README.md` 위치 이동**
   - 메인 화면에 나오도록 위치 이동
