import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# AUTolog hanya untuk BASIC
mlflow.autolog()

df = pd.read_csv("C:\Eksperimen_SML_Anjuan-Kaisar\Membangun_model\heart_disease_preprocessed.csv")

X = df.drop(columns=["Heart Disease Status"])
y = df["Heart Disease Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
