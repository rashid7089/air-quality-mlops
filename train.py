import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# -------------------------------------------------------
# Load processed dataset
# -------------------------------------------------------

df = pd.read_parquet("data/processed/model_table.parquet")

# -------------------------------------------------------
# Split data chronologically
# -------------------------------------------------------

split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

# -------------------------------------------------------
# Prepare features
# -------------------------------------------------------

X_train = train.drop(columns=["high_pollution_next_hour"]).copy()
X_test = test.drop(columns=["high_pollution_next_hour"]).copy()

y_train = train["high_pollution_next_hour"]
y_test = test["high_pollution_next_hour"]

# -------------------------------------------------------
# Remove datetime columns
# -------------------------------------------------------

datetime_columns = X_train.select_dtypes(
    include=["datetime64[ns]", "datetime64"]
).columns

if len(datetime_columns) > 0:
    print("\nDropping datetime columns:")
    print(list(datetime_columns))

    X_train = X_train.drop(columns=datetime_columns)
    X_test = X_test.drop(columns=datetime_columns)

# -------------------------------------------------------
# Verify remaining feature types
# -------------------------------------------------------

print("\nFeature dtypes:")
print(X_train.dtypes)

# -------------------------------------------------------
# MLflow
# -------------------------------------------------------

mlflow.set_experiment("AirQualityPrediction")

# -------------------------------------------------------
# Models
# -------------------------------------------------------

models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}

best_model = None
best_name = None
best_f1 = -1

os.makedirs("models", exist_ok=True)

# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probabilities)
        else:
            auc = 0.5

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("ROC_AUC", auc)

        mlflow.sklearn.log_model(model, name="model")

        print("=" * 60)
        print(name)
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

# -------------------------------------------------------
# Save Best Model
# -------------------------------------------------------

joblib.dump(best_model, "models/model.joblib")

print("\nBest model:", best_name)
print("Best F1:", round(best_f1, 4))
print("Best model saved to models/model.joblib")