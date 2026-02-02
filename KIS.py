import requests
import pandas as pd
import json
import time

# =========================================================
# [설정] 본인의 앱키/시크릿키 입력
# =========================================================
APP_KEY = "PSbk6Xl0PUXbUqVljrmhFwpMg2jBY4DPC41C"
APP_SECRET = "t/pAxqo5un71gPnp2hR6bN580V6Iwc+NEKqHtlp21JuSykAxdAxqUofZNtMeMJqNsIcMWParg5cL4tte/FoZzxDygMB64K98eIhIum3HBEj3KL8BRWZWlagM4siYb0wdX2Gut7z5Mz5V7tzX+EZ747PKM1293NIjLn2XW+4T502Pqqbx1Oo="
URL_BASE = "https://openapi.koreainvestment.com:9443" 

# 토큰 발급
def get_access_token():
    headers = {"content-type": "application/json"}
    body = {"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET}
    res = requests.post(f"{URL_BASE}/oauth2/tokenP", headers=headers, data=json.dumps(body))
    return res.json()['access_token']

# =========================================================
# [핵심 함수] 하루치 1분봉 가져오기
# =========================================================
def get_today_1min_chart(ticker):
    token = get_access_token()
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST03010200",  # 주식당일분봉조회
        "custtype": "P"
    }
    
    all_data = []
    next_time = "153000" # 장 마감 시간부터 역순으로 조회 시작
    
    print(f"🔄 [{ticker}] 1분봉 데이터 수집 시작 (역순 조회)...")

    # 09:00 이전으로 갈 때까지 반복
    while int(next_time) > 90000:
        params = {
            "FID_ETC_CLS_CODE": "",
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_HOUR_1": next_time, 
            "FID_PW_DATA_INCU_YN": "N"  # N: 해당 시간 포함
        }

        res = requests.get(f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
                           headers=headers, params=params)
        
        chunk = res.json().get('output2', [])
        
        if not chunk:
            print("⚠️ 데이터 없음 (장 시작 전이거나 휴장일)")
            break
            
        all_data.extend(chunk)
        
        # 다음 조회를 위해 마지막 시간 갱신
        last_time = chunk[-1]['stck_cntg_hour']
        next_time = last_time
        
        print(f"   Running... {last_time} 데이터까지 수집 완료")
        
        # 09:00:00 데이터가 포함되어 있으면 종료
        if int(last_time) <= 90000:
            break
            
        time.sleep(0.1) # 과부하 방지

    # DataFrame 변환
    df = pd.DataFrame(all_data)
    
    if df.empty:
        return df

    # 컬럼 정리
    clean_cols = {
        'stck_cntg_hour': 'Time',  # 시간
        'stck_prpr': 'Close',      # 현재가(종가)
        'stck_oprc': 'Open',       # 시가
        'stck_hgpr': 'High',       # 고가
        'stck_lwpr': 'Low',        # 저가
        'cntg_vol': 'Volume'       # 거래량
    }
    df = df[clean_cols.keys()].rename(columns=clean_cols)
    
    # 숫자형 변환
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col])
        
    # 시간 오름차순 정렬 (09:00 -> 15:30)
    df = df.sort_values('Time').reset_index(drop=True)
    
    # [필터링] 정규장 시간 (09:00 ~ 15:30)만 남기기
    # 090000 ~ 153000 사이 데이터만 추출 (장전/장후 시간외 거래 제외)
    df = df[(df['Time'] >= "090000") & (df['Time'] <= "153000")]

    return df

# --- 실행 ---
df = get_today_1min_chart("005930") # 삼성전자

print("\n[최종 결과: 삼성전자 1분봉]")
print(df.head()) # 장 시작 직후 (09:00)
print("...")
print(df.tail()) # 장 마감 직전 (15:30)

# 엑셀 저장
df.to_excel("samsung_1min.xlsx", index=False)
print("✅ samsung_1min.xlsx 저장 완료")
# python KIS.py