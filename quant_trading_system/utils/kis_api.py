"""
한국투자증권 Open API 공통 유틸리티
- 토큰 발급/갱신
- 국내 현재가 조회 / 지정가 매수·매도
- 해외 현재가 조회 / 지정가 매수·매도
- 잔고 조회
"""

import time
import json
import logging
import requests
import settings as cfg

logger = logging.getLogger(__name__)
APP_KEY = cfg.KIS_APP_KEY
APP_SECRET = cfg.KIS_APP_SECRET
ACCOUNT_NO = cfg.KIS_ACCOUNT_NO
ACCOUNT_CODE = cfg.KIS_ACCOUNT_CODE
BASE_URL = cfg.KIS_BASE_URL

# 토큰 불러오기
_token_cache = {"token": None, "issued_at": 0}
TOKEN_VALIDITY_SEC = cfg.TOKEN_VALIDITY_SEC


def get_access_token(force_refresh: bool = False) -> str:
    """
    액세스 토큰 발급. 캐시된 토큰이 유효하면 재사용,
    23시간 경과 시 자동 재발급.
    """
    now = time.time()
    if (
        not force_refresh
        and _token_cache["token"]
        and (now - _token_cache["issued_at"]) < TOKEN_VALIDITY_SEC
    ):
        return _token_cache["token"]

    url = f"{BASE_URL}/oauth2/tokenP"
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
    }
    res = requests.post(url, headers=headers, data=json.dumps(body), timeout=10)
    res.raise_for_status()
    token = res.json()["access_token"]

    _token_cache["token"] = token
    _token_cache["issued_at"] = time.time()
    logger.info("KIS 액세스 토큰 발급 완료")
    return token


# 현재가 조회 기능
def get_current_price(stock_code: str) -> int | None:
    token = get_access_token()
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST01010100",
    }
    params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": stock_code}

    try:
        res = requests.get(url, headers=headers, params=params, timeout=cfg.API_TIMEOUT_SEC)
        data = res.json()
        if data["rt_cd"] == "0":
            return int(data["output"]["stck_prpr"])
    except Exception as e:
        logger.error(f"현재가 조회 실패 ({stock_code}): {e}")
    return None


# api 주문
def _place_order(tr_id: str, stock_code: str, qty: int, price: int) -> dict:
    token = get_access_token()
    url = f"{BASE_URL}/uapi/domestic-stock/v1/trading/order-cash"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": tr_id,
        "custtype": "P",
    }
    body = {
        "CANO": ACCOUNT_NO,
        "ACNT_PRDT_CD": ACCOUNT_CODE,
        "PDNO": stock_code,
        "ORD_DVSN": "00",  # 지정가
        "ORD_QTY": str(qty),
        "ORD_UNPR": str(price),
    }
    res = requests.post(url, headers=headers, data=json.dumps(body), timeout=cfg.API_TIMEOUT_SEC)
    result = res.json()
    logger.info(f"주문 결과 [{tr_id}] {stock_code}: {result.get('msg1', '')}")
    return result


def buy_limit_order(stock_code: str, qty: int, price: int) -> dict:
    """지정가 매수 주문 (실전투자)."""
    return _place_order("TTTC0802U", stock_code, qty, price)


def sell_limit_order(stock_code: str, qty: int, price: int) -> dict:
    """지정가 매도 주문 (실전투자)."""
    return _place_order("TTTC0801U", stock_code, qty, price)


#  해외 주식 API -> 해외 주식 용

# 거래소 코드 매핑
EXCHANGE_CODE = {
    "NASD": "NASD",   # 나스닥
    "NYSE": "NYSE",   # 뉴욕
    "AMEX": "AMEX",   # 아멕스
    "SEHK": "SEHK",   # 홍콩
    "SHAA": "SHAA",   # 상해A
    "SZAA": "SZAA",   # 심천A
    "TKSE": "TKSE",   # 도쿄
}


def get_overseas_current_price(exchange: str, stock_code: str) -> float | None:
    """
    해외 주식 현재가 조회.
    """
    token = get_access_token()
    url = f"{BASE_URL}/uapi/overseas-price/v1/quotations/price"
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "HHDFS00000300",
    }
    params = {
        "AUTH": "",
        "EXCD": exchange,
        "SYMB": stock_code,
    }
    try:
        res = requests.get(url, headers=headers, params=params, timeout=cfg.API_TIMEOUT_SEC)
        data = res.json()
        if data["rt_cd"] == "0":
            return float(data["output"]["last"])
    except Exception as e:
        logger.error(f"해외 현재가 조회 실패 ({exchange}:{stock_code}): {e}")
    return None


def _place_overseas_order(tr_id: str, exchange: str, stock_code: str, qty: int, price: float) -> dict:
    """해외 주식 주문 공통."""
    token = get_access_token()
    url = f"{BASE_URL}/uapi/overseas-stock/v1/trading/order"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": tr_id,
        "custtype": "P",
    }
    body = {
        "CANO": ACCOUNT_NO,
        "ACNT_PRDT_CD": ACCOUNT_CODE,
        "OVRS_EXCG_CD": exchange,
        "PDNO": stock_code,
        "ORD_QTY": str(qty),
        "OVRS_ORD_UNPR": str(price),
        "ORD_SVR_DVSN_CD": "0",
        "ORD_DVSN": "00",
    }
    res = requests.post(url, headers=headers, data=json.dumps(body), timeout=cfg.API_TIMEOUT_SEC)
    result = res.json()
    logger.info(f"해외 주문 결과 [{tr_id}] {exchange}:{stock_code}: {result.get('msg1', '')}")
    return result


def buy_overseas_limit_order(exchange: str, stock_code: str, qty: int, price: float) -> dict:
    """해외 주식 지정가 매수 (실전투자)."""
    return _place_overseas_order("JTTT1002U", exchange, stock_code, qty, price)


def sell_overseas_limit_order(exchange: str, stock_code: str, qty: int, price: float) -> dict:
    """해외 주식 지정가 매도 (실전투자)."""
    return _place_overseas_order("JTTT1006U", exchange, stock_code, qty, price)


def get_overseas_balance() -> list[dict] | None:
    """해외 주식 잔고 조회."""
    token = get_access_token()
    url = f"{BASE_URL}/uapi/overseas-stock/v1/trading/inquire-balance"
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "JTTT3012R",
    }
    params = {
        "CANO": ACCOUNT_NO,
        "ACNT_PRDT_CD": ACCOUNT_CODE,
        "OVRS_EXCG_CD": "NASD",
        "TR_CRCY_CD": "USD",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": "",
    }
    try:
        res = requests.get(url, headers=headers, params=params, timeout=cfg.API_TIMEOUT_SEC)
        data = res.json()
        if data["rt_cd"] == "0":
            return data["output1"]
    except Exception as e:
        logger.error(f"해외 잔고 조회 실패: {e}")
    return None


#  국내 잔고 조회  -> 본인 잔고 궁금할 경우
def get_balance() -> list[dict] | None:
    token = get_access_token()
    url = f"{BASE_URL}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "TTTC8434R",
    }
    params = {
        "CANO": ACCOUNT_NO,
        "ACNT_PRDT_CD": ACCOUNT_CODE,
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    try:
        res = requests.get(url, headers=headers, params=params, timeout=cfg.API_TIMEOUT_SEC)
        data = res.json()
        if data["rt_cd"] == "0":
            return data["output1"]
    except Exception as e:
        logger.error(f"잔고 조회 실패: {e}")
    return None
