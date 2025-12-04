from flask import Flask, render_template, request, jsonify, url_for, redirect, send_from_directory
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained models
crop_model = joblib.load("models/crop_model.pkl")
crop_label_encoder = joblib.load("models/crop_label_encoder.pkl")
crop_scaler = joblib.load("models/crop_scaler.pkl")

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/')
def home():
    return render_template('crop_form.html')

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
            "redirect_url": url_for('crop_result', crop=recommended_crop, image_url=image_url, N=N, P=P, K=K)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

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
    return render_template('soil_fertility.html')

if __name__ == '__main__':
    app.run(debug=True)
