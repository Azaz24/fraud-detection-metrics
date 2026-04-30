import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score

#--------------------------------
# Dataset
# -------------------------------
data = {
    'transaction_amount': [200, 4500, 150, 8000, 300, 12000, 500, 95, 7500, 180,
                           250, 9500, 400, 6000, 130, 11000, 350, 75, 8800, 220],
    'login_attempts':     [1, 5, 1, 8, 2, 9, 1, 1, 7, 2,
                           1, 6, 2, 5, 1, 8, 1, 1, 9, 2],
    'account_age_days':   [500, 20, 800, 10, 650, 5, 900, 1200, 15, 700,
                           550, 8, 450, 25, 950, 3, 600, 1100, 12, 480],
    'fraud':              [0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
                           0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
}
df = pd.DataFrame(data)

# -------------------------------
# Features & Target
# -------------------------------
X = df.drop('fraud', axis=1)
y = df['fraud']

# -------------------------------
# Train-Test Split (80/20)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Model Training
# -------------------------------
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# -------------------------------
# Predictions (Default Threshold 0.5)
# -------------------------------
y_pred = model.predict(X_test_scaled)

# -------------------------------
# Metrics
# -------------------------------
print("=== Default Threshold (0.5) ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# Threshold Tuning (0.3)
# -------------------------------
y_probs = model.predict_proba(X_test_scaled)[:, 1]
y_pred_03 = (y_probs >= 0.3).astype(int)

print("\n=== Threshold = 0.3 ===")
print("Precision:", precision_score(y_test, y_pred_03))
print("Recall:", recall_score(y_test, y_pred_03))

# Explanation:
# Lowering threshold from 0.5 to 0.3 increases recall (more fraud cases detected)
# but may decrease precision (more false alarms).
# This is important in fraud detection because catching fraud is more critical
# than avoiding false positives.

# -------------------------------
# ROC-AUC Score
# -------------------------------
roc_score = roc_auc_score(y_test, y_probs)
print("\nROC-AUC Score:", roc_score)

# Explanation:
# ROC-AUC tells how well the model distinguishes between fraud and normal transactions.
# A score closer to 1 means better performance.