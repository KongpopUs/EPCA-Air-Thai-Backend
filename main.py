from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from datetime import timedelta

app = FastAPI()

# CORS สำหรับเชื่อมต่อ frontend Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ถ้าจะจำกัด domain ใส่ URL frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Load model + scalers --------
model = joblib.load("xgboost_model.pkl")
feature_scaler = joblib.load("scaler.pkl")      # temp/humidity/pressure scaler
pm_scaler = joblib.load("pm_scaler.pkl")        # pm2.5 scaler

# -------- Load dataset --------
data_df = pd.read_csv("processed_datapm_2023_rain-1.csv")
data_df["Date"] = pd.to_datetime(data_df["Date"], errors="coerce")
data_df = data_df.dropna(subset=["Date"])
data_df = data_df.sort_values("Date")


# -------- Schema --------
class PM25Request(BaseModel):
    temperature: float
    humidity: float
    pressure: float
    pm2_5_lag1: float
    pm2_5_lag24: float

class DateRequest(BaseModel):
    date: str


# ===================================================================
# 1) CURRENT — ส่งข้อมูลล่าสุดให้ frontend
# ===================================================================
@app.get("/current")
def get_current_pm25():
    latest = data_df.tail(1).iloc[0]

    return {
        "pm25": round(float(latest["pm2_5"]), 2),
        "temperature": round(float(latest["temperature"]), 2),
        "humidity": round(float(latest["humidity"]), 2),
        "pressure": round(float(latest["pressure"]), 2),
        "date": str(latest["Date"])
    }


# ===================================================================
# 2) FORECAST — พยากรณ์ล่วงหน้า (7 วัน)
# ===================================================================
@app.get("/forecast")
def forecast_pm25():

    last_row = data_df.tail(24).copy()

    if len(last_row) < 24:
        raise HTTPException(400, "Not enough data for forecasting")

    forecast_list = []

    for i in range(5):    # 🔥 พยากรณ์แค่ 5 วัน

        # scaling weather
        scaled_weather = feature_scaler.transform(
            last_row[['temperature', 'humidity', 'pressure']]
        )
        temp = scaled_weather[:, 0].mean()
        hum = scaled_weather[:, 1].mean()
        press = scaled_weather[:, 2].mean()

        # scaling PM lags
        scaled_pm = pm_scaler.transform(last_row[['pm2_5']]).flatten()
        lag1 = scaled_pm[-1]
        lag24 = scaled_pm[0]

        # Predict
        df_pred = pd.DataFrame([{
            "temperature": temp,
            "humidity": hum,
            "pressure": press,
            "pm2_5_lag1": lag1,
            "pm2_5_lag24": lag24
        }])

        y_norm = model.predict(df_pred)[0]
        y_real = pm_scaler.inverse_transform([[y_norm]])[0][0]

        forecast_date = (data_df["Date"].max() + timedelta(days=i+1)).strftime("%Y-%m-%d")

        forecast_list.append({
            "date": forecast_date,
            "pm25": round(float(y_real), 2),   # 🔥 ปัดทศนิยม
            "temperature": round(float(last_row["temperature"].mean()), 2),
            "humidity": round(float(last_row["humidity"].mean()), 2),
            "pressure": round(float(last_row["pressure"].mean()), 2)
        })

        # update autoregression
        new_row = {
            "Date": data_df["Date"].max() + timedelta(days=i+1),
            "pm2_5": y_real,
            "temperature": last_row["temperature"].mean(),
            "humidity": last_row["humidity"].mean(),
            "pressure": last_row["pressure"].mean(),
        }
        last_row = pd.concat([last_row.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)

    return forecast_list


# ===================================================================
# 3) Predict PM2.5 จากค่าที่ผู้ใช้ส่งมา (input manual)
# ===================================================================
@app.post("/predict_pm25")
def predict_pm25(req: PM25Request):

    temp_norm, hum_norm, press_norm = feature_scaler.transform(
        [[req.temperature, req.humidity, req.pressure]]
    )[0]

    pm1_norm = pm_scaler.transform([[req.pm2_5_lag1]])[0][0]
    pm24_norm = pm_scaler.transform([[req.pm2_5_lag24]])[0][0]

    df = pd.DataFrame([{
        "temperature": temp_norm,
        "humidity": hum_norm,
        "pressure": press_norm,
        "pm2_5_lag1": pm1_norm,
        "pm2_5_lag24": pm24_norm
    }])

    y_norm = model.predict(df)[0]
    y_real = pm_scaler.inverse_transform([[y_norm]])[0][0]

    return {
        "predicted_pm25_normalized": round(float(y_norm), 2),
        "predicted_pm25_real": round(float(y_real), 2)
    }



# ===================================================================
# 4) Predict PM2.5 จากวันที่เลือก
# ===================================================================
@app.post("/predict_pm25_by_date")
def predict_by_date(req: DateRequest):

    date = pd.to_datetime(req.date)
    past24 = data_df[data_df["Date"] < date].tail(24)

    if len(past24) < 24:
        raise HTTPException(400, "Not enough historical data")

    # ค่าจริงเฉลี่ย 24 ชั่วโมงล่าสุดก่อนวันที่เลือก
    real_temp = round(float(past24["temperature"].mean()), 2)
    real_hum = round(float(past24["humidity"].mean()), 2)
    real_press = round(float(past24["pressure"].mean()), 2)

    # ใช้ค่า normalized สำหรับโมเดล
    scaled_weather = feature_scaler.transform(
        past24[['temperature', 'humidity', 'pressure']]
    )
    temp = scaled_weather[:, 0].mean()
    hum = scaled_weather[:, 1].mean()
    press = scaled_weather[:, 2].mean()

    scaled_pm = pm_scaler.transform(past24[['pm2_5']]).flatten()
    lag1 = scaled_pm[-1]
    lag24 = scaled_pm[0]

    df = pd.DataFrame([{
        "temperature": temp,
        "humidity": hum,
        "pressure": press,
        "pm2_5_lag1": lag1,
        "pm2_5_lag24": lag24
    }])

    y_norm = model.predict(df)[0]
    y_real = pm_scaler.inverse_transform([[y_norm]])[0][0]

    return {
        "date": req.date,
        "predicted_pm25": round(float(y_real), 2),
        "temperature": real_temp,
        "humidity": real_hum,
        "pressure": real_press,
    }

@app.post("/predict_next_days")
def predict_next_days(payload: dict):
    start_date = payload["date"]
    days = payload.get("days", 5)

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}

    results = []
    current_dt = start_dt

    for i in range(1, days + 1):
        predict_day = current_dt + timedelta(days=i)

        df_input = pd.DataFrame([{
            "temperature": 30,      # ใส่ค่าจริงของคุณ
            "pressure": 1012,       # หรือคำนวณจาก historical
            "humidity": 70
        }])

        pm25_pred = float(model.predict(df_input)[0])

        results.append({
            "date": predict_day.strftime("%Y-%m-%d"),
            "pm25": pm25_pred,
            "temperature": df_input["temperature"][0],
            "humidity": df_input["humidity"][0],
            "pressure": df_input["pressure"][0]
        })

    return results