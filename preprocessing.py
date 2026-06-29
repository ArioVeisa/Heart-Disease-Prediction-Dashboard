# =========================
# PHASE 1 & 2
# DATA CHECK + PREPROCESSING
# =========================

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# =========================
# 1. LOAD DATASET
# =========================

df = pd.read_csv("heart.csv")

print("===== 5 DATA TERATAS =====")
print(df.head())

print("\n===== INFO DATASET =====")
print(df.info())

print("\n===== JUMLAH BARIS & KOLOM =====")
print(df.shape)

print("\n===== NAMA KOLOM =====")
print(df.columns.tolist())

# =========================
# 2. CEK MISSING VALUE
# =========================

print("\n===== MISSING VALUE =====")
print(df.isnull().sum())

# =========================
# 3. CEK DUPLIKAT
# =========================

print("\n===== JUMLAH DATA DUPLIKAT =====")
print(df.duplicated().sum())

# =========================
# 4. CEK DISTRIBUSI TARGET
# =========================

print("\n===== DISTRIBUSI TARGET =====")
print(df["HeartDisease"].value_counts())

print("\n===== DISTRIBUSI TARGET (%) =====")
print(df["HeartDisease"].value_counts(normalize=True) * 100)

# =========================
# 5. CEK ANOMALI DATA
# =========================

print("\n===== CEK ANOMALI SEBELUM PREPROCESSING =====")
print("Jumlah Cholesterol = 0:", (df["Cholesterol"] == 0).sum())
print("Jumlah RestingBP = 0:", (df["RestingBP"] == 0).sum())

# =========================
# 6. TANGANI CHOLESTEROL = 0
# =========================

median_cholesterol = df[df["Cholesterol"] != 0]["Cholesterol"].median()
df["Cholesterol"] = df["Cholesterol"].replace(0, median_cholesterol)

print("\n===== CHOLESTEROL SETELAH IMPUTASI =====")
print("Median Cholesterol:", median_cholesterol)
print("Jumlah Cholesterol = 0:", (df["Cholesterol"] == 0).sum())

# =========================
# 7. TANGANI RESTINGBP = 0
# =========================

median_restingbp = df[df["RestingBP"] != 0]["RestingBP"].median()
df["RestingBP"] = df["RestingBP"].replace(0, median_restingbp)

print("\n===== RESTINGBP SETELAH IMPUTASI =====")
print("Median RestingBP:", median_restingbp)
print("Jumlah RestingBP = 0:", (df["RestingBP"] == 0).sum())

# =========================
# 8. LABEL ENCODING
# =========================

categorical_cols = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope"
]

label_mappings = {}

for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    label_mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

print("\n===== HASIL LABEL ENCODING =====")
for col, mapping in label_mappings.items():
    print(f"{col}: {mapping}")

print("\n===== DATASET SETELAH ENCODING =====")
print(df.head())

# =========================
# 9. PISAHKAN FITUR DAN TARGET
# =========================

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

print("\n===== FITUR DAN TARGET =====")
print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# 10. STANDARD SCALER
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n===== STANDARD SCALER SELESAI =====")
print("Mean setelah scaling:", np.round(X_scaled.mean(axis=0), 4))
print("Std setelah scaling:", np.round(X_scaled.std(axis=0), 4))

# =========================
# 11. TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n===== HASIL TRAIN TEST SPLIT =====")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\n===== DISTRIBUSI TARGET TRAIN =====")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True) * 100)

print("\n===== DISTRIBUSI TARGET TEST =====")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True) * 100)

print("\nPHASE 1 & 2 SELESAI ✅")