import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1️⃣ Load dataset tanpa outlier
file_path = "house_pricing_cleaned.csv"  # Ubah sesuai dataset
df_cleaned = pd.read_csv(file_path)

# Pisahkan fitur (X) dan target (Y)
X = df_cleaned.drop(columns=["SalePrice"], errors="ignore")
Y = df_cleaned["SalePrice"]

# 2️⃣ Identifikasi Kolom Kategori (String)
categorical_columns = X.select_dtypes(include=["object"]).columns

# 3️⃣ Konversi Fitur Kategori menjadi Numerik (One-Hot Encoding)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"
)

X_encoded = preprocessor.fit_transform(X)

# Split data menjadi Training (80%) dan Testing (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)

# 4️⃣ Model Linear Regression sebagai baseline
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

Y_pred_linear = lin_reg.predict(X_test)

mse_linear = mean_squared_error(Y_test, Y_pred_linear)
r2_linear = r2_score(Y_test, Y_pred_linear)

print("\n📌 **Hasil Evaluasi Model Linear Regression:**")
print(f"MSE: {mse_linear:.2f}")
print(f"R²: {r2_linear:.4f}")

# 5️⃣ Model Polynomial Regression (Degree 2 & 3)
for degree in [2, 3]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, Y_train)

    Y_pred_poly = poly_reg.predict(X_test_poly)

    mse_poly = mean_squared_error(Y_test, Y_pred_poly)
    r2_poly = r2_score(Y_test, Y_pred_poly)

    print(f"\n📌 **Hasil Evaluasi Model Polynomial Regression (Degree {degree}):**")
    print(f"MSE: {mse_poly:.2f}")
    print(f"R²: {r2_poly:.4f}")

    # 6️⃣ Visualisasi Hasil Prediksi
    plt.figure(figsize=(6, 5))
    plt.scatter(Y_test, Y_pred_poly, alpha=0.5, label=f"Degree {degree}")
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "--", color="red")
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title(f"📌 Polynomial Regression (Degree {degree})")
    plt.legend()
    plt.savefig(f"polynomial_regression_degree_{degree}.png", bbox_inches="tight")
    plt.show()

    print(f"📂 Visualisasi telah disimpan sebagai 'polynomial_regression_degree_{degree}.png' 🎯")
