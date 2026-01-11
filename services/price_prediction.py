# services/predictfile.py
import pandas as pd
from sklearn.linear_model import LinearRegression

CSV_PATH = "dataset/market_prices.csv"

def predict_price(crop, district):
    df = pd.read_csv(CSV_PATH)

    df = df[(df["crop"] == crop) & (df["district"] == district)]
    df = df.sort_values("date")

    if len(df) < 2:
        return None

    df = df.reset_index(drop=True)
    df["day_index"] = df.index

    X = df[["day_index"]]
    y = df["price"]

    model = LinearRegression()
    model.fit(X, y)

    next_day_index = [[df["day_index"].max() + 1]]
    predicted_price = model.predict(next_day_index)[0]

    return round(float(predicted_price), 2)
