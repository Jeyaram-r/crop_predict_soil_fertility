import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv(
    r"D:\Final year project\crop_model\dataset\crop_recommendation.csv"
)

# Features and target
X = df.drop(columns=["label"])
y = df["label"]

# ==============================
# LABEL ENCODING
# ==============================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

joblib.dump(label_encoder, "models/crop_label_encoder.pkl")

# ==============================
# FEATURE SCALING
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "models/crop_scaler.pkl")

# ==============================
# TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# ==============================
# TRAIN RANDOM FOREST MODEL
# ==============================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# ==============================
# MODEL EVALUATION
# ==============================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

# ==============================
# FEATURE IMPORTANCE (EXPLAINABILITY)
# ==============================

feature_names = X.columns
importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance_df)

# Save feature importance for Flask
joblib.dump(feature_importance_df, "models/feature_importance.pkl")

# ==============================
# SAVE TRAINED MODEL
# ==============================

joblib.dump(model, "models/crop_model.pkl")

print("\n✅ Model training complete.")
print("✅ Files saved in 'models/' directory.")
