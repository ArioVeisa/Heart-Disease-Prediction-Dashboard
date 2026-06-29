import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("heart.csv")

# =========================
# PREPROCESSING
# =========================

median_cholesterol = df[df["Cholesterol"] != 0]["Cholesterol"].median()
df["Cholesterol"] = df["Cholesterol"].replace(0, median_cholesterol)

median_restingbp = df[df["RestingBP"] != 0]["RestingBP"].median()
df["RestingBP"] = df["RestingBP"].replace(0, median_restingbp)

categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# MODEL NAIVE BAYES
# =========================

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# EVALUATION
# =========================

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("=" * 60)
print("EVALUASI MODEL GAUSSIAN NAIVE BAYES")
print("=" * 60)

print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nDetail Confusion Matrix:")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("\nMetrik Evaluasi:")
print(f"Accuracy  : {accuracy:.4f} / {accuracy * 100:.2f}%")
print(f"Precision : {precision:.4f} / {precision * 100:.2f}%")
print(f"Recall    : {recall:.4f} / {recall * 100:.2f}%")
print(f"F1-Score  : {f1:.4f} / {f1 * 100:.2f}%")
print(f"ROC AUC   : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
