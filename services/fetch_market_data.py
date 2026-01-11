import requests
import pandas as pd
from datetime import datetime
import os

CSV_PATH = "dataset/tamilnadu_market_prices.csv"
API_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

PARAMS = {
    "format": "json",
    "limit": 500,
    "filters[state]": "Tamil Nadu"
}

def fetch_and_update_csv():
    try:
        response = requests.get(API_URL, params=PARAMS, timeout=30)

        if response.status_code != 200:
            print("API failed:", response.status_code)
            return

        records = response.json().get("records", [])
        if not records:
            print("No records received")
            return

        df = pd.DataFrame(records)

        df = df.rename(columns={
            "commodity": "crop",
            "market": "market",
            "district": "district",
            "modal_price": "price",
            "arrival_date": "date"
        })

        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["price", "date"])

        os.makedirs("dataset", exist_ok=True)
        df.to_csv(CSV_PATH, index=False)

        print(f"[{datetime.now()}] Market data updated")

    except Exception as e:
        print("Error fetching market data:", e)
        print("Using existing CSV if available")
