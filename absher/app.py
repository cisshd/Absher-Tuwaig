from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI()

# ----------------------------------------------------
#  LOAD DATA
# ----------------------------------------------------
df = pd.read_csv("synthetic_abshar_events_fixed.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

user_profiles = df.groupby("user_id").agg({
    "location_lat": ["mean", "std"],
    "location_lon": ["mean", "std"],
    "login_count_day": ["mean", "std"],
}).reset_index()

user_profiles.columns = [
    "user_id",
    "lat_mean", "lat_std",
    "lon_mean", "lon_std",
    "login_mean", "login_std"
]
user_profiles = user_profiles.replace({0: 1e-6})

# ----------------------------------------------------
#  MODEL CLASS
# ----------------------------------------------------
class Event(BaseModel):
    user_id: str
    location_lat: float
    location_lon: float
    login_count_day: int

def predict_user_event(event):
    uid = event["user_id"]

    if uid not in user_profiles["user_id"].values:
        return {"error": "user_not_found"}

    profile = user_profiles[user_profiles["user_id"] == uid].iloc[0]

    lat_diff = abs(event["location_lat"] - profile["lat_mean"]) / profile["lat_std"]
    lon_diff = abs(event["location_lon"] - profile["lon_mean"]) / profile["lon_std"]
    login_diff = abs(event["login_count_day"] - profile["login_mean"]) / profile["login_std"]

    anomaly_score = lat_diff + lon_diff + login_diff
    threshold = 3.5

    if anomaly_score > threshold:
        risk = "high"
    elif anomaly_score > threshold * 0.6:
        risk = "medium"
    else:
        risk = "normal"

    return {
        "user_id": uid,
        "anomaly_score": float(anomaly_score),
        "risk_level": risk,
        "details": {
            "lat_diff": float(lat_diff),
            "lon_diff": float(lon_diff),
            "login_diff": float(login_diff),
        }
    }

# ----------------------------------------------------
#  API ENDPOINT
# ----------------------------------------------------
@app.post("/predict")
def predict_api(event: Event):
    return predict_user_event(event.dict())
