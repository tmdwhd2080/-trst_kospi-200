import requests
import pandas as pd
import json

# 1. 내 계좌/API 정보 입력 (필수)
APP_KEY = "PSbk6Xl0PUXbUqVljrmhFwpMg2jBY4DPC41C"
APP_SECRET = "t/pAxqo5un71gPnp2hR6bN580V6Iwc+NEKqHtlp21JuSykAxdAxqUofZNtMeMJqNsIcMWParg5cL4tte/FoZzxDygMB64K98eIhIum3HBEj3KL8BRWZWlagM4siYb0wdX2Gut7z5Mz5V7tzX+EZ747PKM1293NIjLn2XW+4T502Pqqbx1Oo="
URL_BASE = "https://openapi.koreainvestment.com:9443" # 실전투자

# 2. 토큰 발행 (API 사용권한 획득)
def get_access_token():
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET
    }
    res = requests.post(f"{URL_BASE}/oauth2/tokenP", headers=headers, data=json.dumps(body))
    return res.json()['access_token']

# 3. 외국인 순매수 데이터 가져오기 함수
def get_foreign_net_buy(ticker):
    token = get_access_token()
    
    # API 요청 헤더 설정
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST01010900" # [투자자별 일별매매상위] 거래 ID
    }
    
    # API 요청 파라미터
    params = {
        "FID_COND_MRKT_DIV_CODE": "J", # J: 주식
        "FID_INPUT_ISCD": ticker       # 종목코드 (예: 005930)
    }
    
    # 데이터 요청 (GET)
    res = requests.get(f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-investor", 
                       headers=headers, params=params)
    
    # 결과 처리 (JSON -> DataFrame)
    data = res.json().get('output', [])
    
    if data:
        df = pd.DataFrame(data)
        # 필요한 컬럼만 선택해서 깔끔하게 정리
        # stck_bsop_date: 날짜, frgn_ntby_qty: 외국인 순매수량
        df = df[['stck_bsop_date', 'frgn_ntby_qty', 'orgn_ntby_qty', 'prsn_ntby_qty']]
        df.columns = ['날짜', '외국인순매수', '기관순매수', '개인순매수']
        
        # 엑셀로 저장
        filename = f"{ticker}_수급데이터.xlsx"
        df.to_excel(filename, index=False)
        print(f"✅ {filename} 저장 완료!")
        return df
    else:
        print("❌ 데이터를 가져오지 못했습니다.")
        return pd.DataFrame()

# --- 실행 ---
df = get_foreign_net_buy("005930") # 삼성전자
print(df.head())