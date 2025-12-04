import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("D:\Final year project\crop_model\dataset\crop_recommendation.csv")


X = df.drop(columns=["label"])  
y = df["label"]  


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

joblib.dump(label_encoder, "models/crop_label_encoder.pkl")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


joblib.dump(scaler, "models/crop_scaler.pkl")


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


joblib.dump(model, "models/crop_model.pkl")

print("Model training complete. Files saved in 'models/' directory.")
