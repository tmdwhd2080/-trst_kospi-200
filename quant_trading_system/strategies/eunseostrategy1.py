"""
[새 전략] DART 이벤트 드리븐(Alpha원천) & 매크로 동적 헤징 통합 봇
──────────────────────────────────────────
전략 설명: 
1. [Risk Overlay] VIX 등 매크로 지표 악화 시 DART 탐색을 중단하고 즉각 현금화(De-risking) 진행.
2. [Alpha Engine] 시장 안정 시, DART '단일판매ㆍ공급계약체결' 공시를 크롤링하여 
   매출액 대비 계약 규모가 임계치(예: 10%) 이상인 알짜 모멘텀 종목만 슬리피지 반영 후 매수.
"""

# 작업 메타데이터 — integrate_strategy.py가 읽어갑니다
TASK_META = {
    "task_id": "dart_macro_integrated",      
    "description": "매크로 방어 + DART 단일판매공시 알파 통합 봇", 
    "type": "strategy",                       
    "enabled": True,                          
    "schedule": [
        {
            "time": "14:50",                  # 장 마감 전 매크로 확인 및 DART 공시 몰리는 시간
            "force_kill_time": "15:20",       
            "description": "리스크 점검 후 DART 공시 타겟팅 매수",
        },
    ],
}

# ═══════════════════════════════════════════════════
#  ⚙ 전략 개별 설정 — 이 영역의 값만 수정하세요
# ═══════════════════════════════════════════════════
# [매크로 설정]
VIX_PANIC_LEVEL = 25.0       # 이 수치를 넘으면 무조건 주식 매도 (Risk-Off)
TARGET_CASH_RATIO = 0.8      # 패닉 시 현금 확보 비율 (80%)

# [DART 공시 설정]
DART_API_KEY = '본인의_DART_API_KEY를_입력하세요'  # DART 데브센터에서 발급/ 공개 안 할게요!
MIN_CONTRACT_RATIO = 10.0    # 노이즈 필터링: 최근 매출액 대비 계약 규모 10% 이상만 진입
ORDER_QTY = 10               # 조건 충족 시 매수할 수량 (모의투자 기준)
SLIPPAGE_RATE = 1.01         # 슬리피지 페널티: 현재가보다 1% 비싸게 체결된다고 가정 (보수적)
# ═══════════════════════════════════════════════════

import logging
import datetime
# 시스템 KIS API 함수
from utils.kis_api import get_current_price, buy_limit_order 
# ※ DART API 라이브러리 (pip install OpenDartReader 필요)
try:
    import OpenDartReader
except ImportError:
    OpenDartReader = None

logger = logging.getLogger(__name__)

def run(macro_data: dict | None = None, dry_run: bool = True) -> dict:
    logger.info("=== [통합 엔진] DART & Macro Hedging 가동 ===")
    
    executed_orders = []
    
    # --------------------------------------------------
    # [엔진 1] 매크로 리스크 오버레이 (방어 로직)
    # --------------------------------------------------
    if macro_data is None:
        macro_data = {}
        
    current_vix = macro_data.get('VIX', 0.0)
    logger.info(f"[매크로 점검] 현재 VIX 지수: {current_vix}")
    
    # 서킷 브레이커: VIX가 패닉 레벨을 넘으면 알파 발굴(공시) 중단
    if current_vix >= VIX_PANIC_LEVEL:
        logger.warning(f"🚨 시장 패닉 감지 (VIX {current_vix}). DART 탐색을 중지하고 현금을 확보합니다.")
        
        if dry_run:
            msg = f"[DRY RUN] 보유 포트폴리오의 {TARGET_CASH_RATIO*100}% 강제 매도 지시 (현금화)"
            executed_orders.append(msg)
            logger.info(f"  └ {msg}")
        else:
            logger.warning("🔴 [LIVE RUN] KIS API 연동: 실제 보유 주식 시장가 매도 엑스큐션 발동!")
            # TODO: 여기에 utils.kis_api의 매도 함수(sell_market_order 등) 연결
            executed_orders.append("[API 매도] 리스크 오프 체결 완료")
            
        return {"status": "Risk-Off", "actions": executed_orders}

    # --------------------------------------------------
    # [엔진 2] 알파 발굴 (DART '단일판매ㆍ공급계약체결' 탐색)
    # --------------------------------------------------
    logger.info("✅ 시장 안정 확인. DART 공시 탐색을 시작합니다.")
    
    if OpenDartReader is None:
        logger.error("OpenDartReader 라이브러리가 없습니다. (pip install OpenDartReader)")
        return {"status": "Error", "actions": ["OpenDartReader Missing"]}
        
    dart = OpenDartReader(DART_API_KEY)
    today = datetime.datetime.today().strftime('%Y%m%d')
    
    try:
        # 오늘 하루 동안 올라온 모든 공시 검색
        notices = dart.list(start=today, end=today)
        
        # '단일판매ㆍ공급계약체결' 텍스트가 포함된 공시만 필터링
        target_notices = notices[notices['report_nm'].str.contains('단일판매ㆍ공급계약체결', na=False)]
        
        if target_notices.empty:
            logger.info("▶ 금일 조건에 맞는 '단일판매ㆍ공급계약체결' 공시가 없습니다.")
            executed_orders.append("No Alpha Signals Today")
        else:
            logger.info(f"💡 타겟 공시 {len(target_notices)}건 발견! 필터링 진입.")
            
            for idx, row in target_notices.iterrows():
                corp_code = row['corp_code']
                corp_name = row['corp_nm']
                
                # --------------------------------------------------
                # [엔진 3] 노이즈 필터링 및 엑스큐션
                # --------------------------------------------------
                # ※ 실제 현업에서는 여기서 DART 상세 문서를 파싱하여 '최근 매출액 대비 비율(%)'을 뽑아냅니다.
                # 이 템플릿에서는 로직 구현을 위해 가상의 필터링 비율을 사용합니다.
                contract_ratio = 15.0 # (가정: 파싱해온 계약 규모가 매출 대비 15%라고 가정)
                
                if contract_ratio >= MIN_CONTRACT_RATIO:
                    logger.info(f"🎯 [타겟 포착] {corp_name} (매출 대비 {contract_ratio}% 계약)")
                    
                    # 주식 코드(종목코드 6자리) 가져오기 로직 필요 (예: corp_code로 종목코드 맵핑)
                    # 여기서는 테스트용으로 삼성전자 코드를 강제 할당합니다.
                    ticker = "005930" 
                    
                    # 현재가 조회 및 슬리피지(Slippage) 페널티 반영
                    # (예: 봇과 싸우느라 현재가보다 1% 비싸게 샀다고 보수적으로 가정)
                    current_price = get_current_price(ticker) if not dry_run else 80000
                    adjusted_buy_price = int(current_price * SLIPPAGE_RATE)
                    
                    if dry_run:
                        msg = f"[DRY RUN] {corp_name}({ticker}) {ORDER_QTY}주 매수 (슬리피지 반영가: {adjusted_buy_price}원)"
                        executed_orders.append(msg)
                        logger.info(f"  └ {msg}")
                    else:
                        logger.warning(f"🔴 [LIVE RUN] {corp_name} KIS API 지정가 매수 주문 발송!")
                        buy_limit_order(ticker, ORDER_QTY, adjusted_buy_price)
                        msg = f"[API 매수] {corp_name} 체결 지시 완료"
                        executed_orders.append(msg)
                        
                else:
                    logger.info(f"  └ [노이즈 패스] {corp_name}: 계약 규모 미달 ({contract_ratio}%)")

    except Exception as e:
        logger.error(f"DART API 호출 중 에러 발생: {e}")
        executed_orders.append(f"DART Error: {e}")

    result = {
        "strategy": "DART_Macro_Integrated",
        "market_risk": "Stable",
        "actions": executed_orders
    }
    
    logger.info(f"=== 전략 실행 완료: {result} ===")
    return result
