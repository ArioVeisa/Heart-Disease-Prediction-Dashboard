import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("heart.csv")

# =====================================
# HANDLE CHOLESTEROL = 0
# =====================================

median_cholesterol = df[df["Cholesterol"] != 0]["Cholesterol"].median()
df["Cholesterol"] = df["Cholesterol"].replace(0, median_cholesterol)

# =====================================
# HANDLE RESTINGBP = 0
# =====================================

median_restingbp = df[df["RestingBP"] != 0]["RestingBP"].median()
df["RestingBP"] = df["RestingBP"].replace(0, median_restingbp)

# =====================================
# LABEL ENCODING
# =====================================

categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

# =====================================
# SPLIT FEATURE & TARGET
# =====================================

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# =====================================
# STANDARD SCALER
# =====================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================
# GAUSSIAN NAIVE BAYES
# =====================================

model = GaussianNB()

model.fit(X_train, y_train)

# =====================================
# PREDIKSI
# =====================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)

# =====================================
# OUTPUT
# =====================================

print("=" * 60)
print("IMPLEMENTASI GAUSSIAN NAIVE BAYES")
print("=" * 60)

print()

print("Jumlah Data Training :", X_train.shape[0])
print("Jumlah Data Testing  :", X_test.shape[0])

print()

print("Jumlah Feature :", X_train.shape[1])

print()

print("Class Prior Probability")
print(model.class_prior_)

print()

print("Class Labels")
print(model.classes_)

print()

print("5 Prediksi Pertama")
print(y_pred[:5])

print()

print("5 Probabilitas Prediksi Pertama")
print(y_prob[:5])

print()

print("Model Gaussian Naive Bayes berhasil dibuat.")
