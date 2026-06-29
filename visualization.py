import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score
)

# ==========================
# CREATE OUTPUT FOLDER
# ==========================

os.makedirs("output", exist_ok=True)

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("heart.csv")

# ==========================
# PREPROCESSING
# ==========================

median_chol = df[df["Cholesterol"] != 0]["Cholesterol"].median()
df["Cholesterol"] = df["Cholesterol"].replace(0, median_chol)

median_bp = df[df["RestingBP"] != 0]["RestingBP"].median()
df["RestingBP"] = df["RestingBP"].replace(0, median_bp)

categorical_cols = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# TRAIN MODEL
# ==========================

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ==========================
# CONFUSION MATRIX
# ==========================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Heart Disease"]
)

disp.plot()

plt.title("Confusion Matrix - Gaussian Naive Bayes")

plt.savefig(
    "output/confusion_matrix_nb.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================
# ROC CURVE
# ==========================

RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.title("ROC Curve - Gaussian Naive Bayes")

plt.savefig(
    "output/roc_curve_nb.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================
# BAR CHART METRICS
# ==========================

logistic = [
    84.78,
    84.91,
    88.24,
    86.54
]

naive = [
    87.50,
    88.35,
    89.22,
    88.78
]

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

x = range(len(metrics))

width = 0.35

plt.figure(figsize=(8,5))

plt.bar(
    [i-width/2 for i in x],
    logistic,
    width,
    label="Logistic Regression"
)

plt.bar(
    [i+width/2 for i in x],
    naive,
    width,
    label="Naive Bayes"
)

plt.xticks(x, metrics)

plt.ylabel("Score (%)")

plt.title("Model Performance Comparison")

plt.legend()

plt.savefig(
    "output/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("="*50)
print("VISUALIZATION COMPLETED")
print("="*50)
print("output/confusion_matrix_nb.png")
print("output/roc_curve_nb.png")
print("output/model_comparison.png")