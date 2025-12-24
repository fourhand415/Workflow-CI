# ===== IMPORT =====
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ===== EXPERIMENT =====
mlflow.set_experiment("CI-Retraining")

# ===== AUTOLOGGING (WAJIB, SESUAI RUBRIK) =====
mlflow.sklearn.autolog()

# ===== LOAD DATA =====
df = pd.read_csv("heart_disease_preprocessing.csv")

X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== TRAIN =====
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
