import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# LOAD DATA
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "heart_disease_preprocessed.csv")

df = pd.read_csv(DATA_PATH)

label_col = df.columns[-1]
X = df.drop(label_col, axis=1)
y = df[label_col]

# ======================
# DAGSHUB TRACKING
# ======================
mlflow.set_tracking_uri("https://dagshub.com/anjuan14/heart-disease-mlflow.mlflow")
mlflow.set_experiment("Heart Disease Modelling - Advance")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, None],
    "min_samples_split": [2, 5]
}

model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

with mlflow.start_run():
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # ===== MANUAL LOGGING 
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    #  ARTEFAK 
    joblib.dump(best_model, "best_model.pkl")
    mlflow.log_artifact("best_model.pkl")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    # ===== ARTEFAK TAMBAHAN (ADVANCE) =====
    X.describe().to_csv("feature_summary.csv")
    mlflow.log_artifact("feature_summary.csv")

    print("ADVANCE RUN SUCCESS")
    print("F1:", f1)
