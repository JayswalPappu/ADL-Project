import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data_set_for_AD_LAB.csv")

# Drop unnecessary columns
df = df.drop(["TransactionID", "UserID", "DeviceID", "IPAddress", "PhoneNumber"], axis=1)

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
df = df.drop("Timestamp", axis=1)

# Convert TransactionFrequency to numeric
df[["TransactionFrequency", "Period"]] = df["TransactionFrequency"].str.split("/", expand=True)
df["TransactionFrequency"] = df["TransactionFrequency"].astype(int)
df = df.drop("Period", axis=1)

# Convert boolean columns
bool_cols = ["UnusualLocation", "UnusualAmount", "NewDevice", "FraudFlag"]
df[bool_cols] = df[bool_cols].astype(int)

# Feature engineering: Amount ratio
df["AmountRatio"] = df["Amount"] / df["AvgTransactionAmount"]

# Separate features and target
X = df.drop("FraudFlag", axis=1)
y = df["FraudFlag"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Define categorical and numerical features
categorical_features = ["MerchantCategory", "TransactionType", "BankName"]
numerical_features = ["Amount", "Latitude", "Longitude", "AvgTransactionAmount",
                      "TransactionFrequency", "FailedAttempts", "Hour", "DayOfWeek", "AmountRatio"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Apply SMOTE for class imbalance
X_train_preprocessed = preprocessor.fit_transform(X_train)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_preprocessed, y_train)

# Train model
model = RandomForestClassifier(class_weight="balanced", random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5]
}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring="recall")
grid_search.fit(X_train_res, y_train_res)

# Best model
best_model = grid_search.best_estimator_

# Evaluate
X_test_preprocessed = preprocessor.transform(X_test)
y_proba = best_model.predict_proba(X_test_preprocessed)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
optimal_threshold = 0.5  # You can fine-tune this based on precision-recall analysis

print(f"ROC AUC Score: {roc_auc:.2f}")

# Save model and preprocessor
joblib.dump(best_model, "fraud_detection_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(optimal_threshold, "optimal_threshold.pkl")

print("Model and preprocessor saved successfully!")
