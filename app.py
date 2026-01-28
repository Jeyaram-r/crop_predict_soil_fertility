from flask import Flask, render_template, request, jsonify, url_for, redirect, send_from_directory, session
import numpy as np
import joblib
import os
import json
import pandas as pd
from services.price_prediction import predict_price
from apscheduler.schedulers.background import BackgroundScheduler
from services.fetch_market_data import fetch_and_update_csv


from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    fetch_and_update_csv()
    scheduler = BackgroundScheduler()
    scheduler.add_job(fetch_and_update_csv, "interval", hours=24)
    scheduler.start()

app.secret_key = "final_year_secret_key"

# -------------------------
# LOAD ML MODELS
# -------------------------
crop_model = joblib.load("models/crop_model.pkl")
crop_label_encoder = joblib.load("models/crop_label_encoder.pkl")
crop_scaler = joblib.load("models/crop_scaler.pkl")
CSV_PATH = "dataset/tamilnadu_market_prices.csv"
ROADMAP_PATH = os.path.join("dataset", "crop_roadmaps.json")

with open(ROADMAP_PATH, "r") as f:
    CROP_ROADMAPS = json.load(f)
@app.route("/get_crop_roadmap/<crop_name>")
def get_crop_roadmap(crop_name):
    crop_name = crop_name.lower()

    for crop in CROP_ROADMAPS:
        if crop["crop"].lower() == crop_name:
            return crop  # Flask auto-converts dict → JSON

    return {"error": "Crop roadmap not found"}, 404


def load_market_data():
    df = pd.read_csv(CSV_PATH)
    df["price"] = df["price"].astype(int)
    return df
def add_price_trend(df):
    df = df.sort_values(["crop", "market", "date"])

    df["trend"] = "same"
    df["prev_price"] = df.groupby(
        ["crop", "market"]
    )["price"].shift(1)

    df.loc[df["price"] > df["prev_price"], "trend"] = "up"
    df.loc[df["price"] < df["prev_price"], "trend"] = "down"

    return df
@app.route("/api/market-prices")
def market_prices_api():
    df = load_market_data()
    df = add_price_trend(df)

    df["date"] = df["date"].astype(str).str.strip()
    latest_date = df["date"].max()
    df = df[df["date"] == latest_date]

    # 🔥 CRITICAL FIX: remove NaN-causing column
    df = df.drop(columns=["prev_price"], errors="ignore")

    return jsonify({
        "last_updated": latest_date,
        "data": df.to_dict(orient="records")
    })
@app.route("/api/recommend-fertilizer", methods=["POST"])
def recommend_fertilizer():
    data = request.json

    crop = data["crop"]
    N = float(data["N"])
    P = float(data["P"])
    K = float(data["K"])

    crop_req = {
        "rice": {"N": 120, "P": 60, "K": 40},
        "wheat": {"N": 100, "P": 50, "K": 40},
        "maize": {"N": 150, "P": 75, "K": 50},
        "apple": {"N": 70, "P": 35, "K": 70}
    }

    req = crop_req[crop.lower()]

    N_def = max(req["N"] - N, 0)
    P_def = max(req["P"] - P, 0)
    K_def = max(req["K"] - K, 0)

    urea = round((N_def / 0.46) / 2.47, 2)
    dap  = round((P_def / 0.46) / 2.47, 2)
    mop  = round((K_def / 0.60) / 2.47, 2)

    return jsonify({
        "fertilizers": {
            "Urea (kg/acre)": urea,
            "DAP (kg/acre)": dap,
            "MOP (kg/acre)": mop
        },
        "application_stage": "Basal + Top Dressing"
    })

@app.route("/api/predict-price")
def predict_price():
    # load trained model
    # predict next price
    pass
@app.route("/api/predict-price")
def predict_price_api():
    crop = request.args.get("crop")
    district = request.args.get("district")

    if not crop or not district:
        return jsonify({"error": "crop and district required"}), 400

    predicted = predict_price(crop, district)

    if predicted is None:
        return jsonify({"error": "Not enough data"}), 404

    return jsonify({
        "crop": crop,
        "district": district,
        "predicted_price": predicted
    })
@app.route("/visitor/market-price")
def visitor_market_price():
    return render_template("visitor/market_price.html")

# -------------------------
# SIMPLE USER STORAGE (FOR PROJECT)
# -------------------------
users = []  # [{name,email,password,role}]

# -------------------------
# STATIC FILES
# -------------------------
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# -------------------------
# LANDING PAGE
# -------------------------
@app.route('/')
def landing():
    return render_template('auth/landing.html')

# -------------------------
# SIGNUP
# -------------------------
@app.route('/signup/<role>', methods=['GET', 'POST'])
def signup(role):
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        users.append({
            "name": name,
            "email": email,
            "password": password,
            "role": role
        })

        return redirect(url_for('login', role=role))

    return render_template('auth/signup.html', role=role)

# -------------------------
# LOGIN
# -------------------------
@app.route('/login/<role>', methods=['GET', 'POST'])
def login(role):
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        for user in users:
            if user['email'] == email and user['role'] == role:
                if check_password_hash(user['password'], password):
                    session['user'] = user['email']
                    session['role'] = role

                    if role == 'farmer':
                        return redirect(url_for('crop_form'))
                    else:
                        return redirect(url_for('market_price'))

        return "Invalid credentials", 401

    return render_template('auth/login.html', role=role)

# -------------------------
# LOGOUT
# -------------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

# -------------------------
# FARMER PAGES
# -------------------------
@app.route('/crop-form')
def crop_form():
    if session.get('role') != 'farmer':
        return redirect(url_for('landing'))
    return render_template('crop_form.html')

@app.route('/crop_result')
def crop_result():
    crop = request.args.get('crop', 'Unknown Crop')
    image_url = request.args.get('image_url', '')
    N = request.args.get('N')
    P = request.args.get('P')
    K = request.args.get('K')
    return render_template('crop_result.html', crop=crop, image_url=image_url, N=N, P=P, K=K)

@app.route('/predict_fertility')
def predict_fertility():
    if session.get('role') != 'farmer':
        return redirect(url_for('landing'))
    return render_template('soil_fertility.html')

# -------------------------
# VISITOR PAGE
# -------------------------
@app.route('/market-price')
def market_price():
    if session.get('role') != 'visitor':
        return redirect(url_for('landing'))

    prices = [
        {"crop": "Rice", "market": "Chennai", "price": 42, "date": "Today"},
        {"crop": "Wheat", "market": "Coimbatore", "price": 38, "date": "Today"},
        {"crop": "Maize", "market": "Madurai", "price": 30, "date": "Today"}
    ]

    return render_template('visitor/market_price.html', prices=prices)

# -------------------------
# ML API (UNCHANGED)
# -------------------------
@app.route('/api/recommend_crop', methods=['POST'])
def recommend_crop():
    try:
        data = request.json

        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        pH = float(data['pH'])
        rainfall = float(data['rainfall'])

        input_features = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
        input_features_scaled = crop_scaler.transform(input_features)

        predicted_label = crop_model.predict(input_features_scaled)[0]
        recommended_crop = crop_label_encoder.inverse_transform([predicted_label])[0]

        image_filename = f"{recommended_crop.lower()}.jpg"
        image_path = os.path.join("static/crop_images", image_filename)

        if not os.path.exists(image_path):
            image_filename = "not_found.jpg"

        image_url = url_for('serve_static', filename=f'crop_images/{image_filename}', _external=True)

        return jsonify({
            "redirect_url": url_for(
                'crop_result',
                crop=recommended_crop,
                image_url=image_url,
                N=N, P=P, K=K
            )
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -------------------------
# RUN
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, render_template, request, jsonify, url_for, redirect, send_from_directory
# import numpy as np
# import joblib
# import os

# app = Flask(__name__)

# # Load trained models
# crop_model = joblib.load("models/crop_model.pkl")
# crop_label_encoder = joblib.load("models/crop_label_encoder.pkl")
# crop_scaler = joblib.load("models/crop_scaler.pkl")

# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     return send_from_directory('static', filename)

# @app.route('/')
# def home():
#     return render_template('crop_form.html')

# @app.route('/api/recommend_crop', methods=['POST'])
# def recommend_crop():
#     try:
#         data = request.json  

#         N = float(data['N'])
#         P = float(data['P'])
#         K = float(data['K'])
#         temperature = float(data['temperature'])
#         humidity = float(data['humidity'])
#         pH = float(data['pH'])
#         rainfall = float(data['rainfall'])

#         input_features = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
#         input_features_scaled = crop_scaler.transform(input_features)

#         predicted_label = crop_model.predict(input_features_scaled)[0]
#         recommended_crop = crop_label_encoder.inverse_transform([predicted_label])[0]

#         image_filename = f"{recommended_crop.lower()}.jpg"
#         image_path = os.path.join("static/crop_images", image_filename)

#         if not os.path.exists(image_path):
#             image_filename = "not_found.jpg"

#         image_url = url_for('serve_static', filename=f'crop_images/{image_filename}', _external=True)

#         return jsonify({
#             "redirect_url": url_for('crop_result', crop=recommended_crop, image_url=image_url, N=N, P=P, K=K)
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# @app.route('/crop_result')
# def crop_result():
#     crop = request.args.get('crop', 'Unknown Crop')
#     image_url = request.args.get('image_url', '')
#     N = request.args.get('N')
#     P = request.args.get('P')
#     K = request.args.get('K')
#     return render_template('crop_result.html', crop=crop, image_url=image_url, N=N, P=P, K=K)

# @app.route('/predict_fertility')
# def predict_fertility():
#     return render_template('soil_fertility.html')

# if __name__ == '__main__':
#     app.run(debug=True)
