import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1️⃣ Load dataset tanpa outlier
file_path = "house_pricing_cleaned.csv"  # Ubah sesuai dataset
df_cleaned = pd.read_csv(file_path)

# Pisahkan fitur (X) dan target (Y)
X = df_cleaned.drop(columns=["SalePrice"], errors="ignore")
Y = df_cleaned["SalePrice"]

# 2️⃣ Identifikasi Kolom Kategori
categorical_columns = X.select_dtypes(include=["object"]).columns

# 3️⃣ Konversi Fitur Kategori menjadi Numerik (One-Hot Encoding) + Scaling
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("num", StandardScaler(), X.select_dtypes(include=["number"]).columns)
    ]
)

X_encoded = preprocessor.fit_transform(X)

# Split data menjadi Training (80%) dan Testing (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)

# 4️⃣ Linear Regression sebagai baseline
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
Y_pred_linear = lin_reg.predict(X_test)

mse_linear = mean_squared_error(Y_test, Y_pred_linear)
r2_linear = r2_score(Y_test, Y_pred_linear)

# 5️⃣ Polynomial Regression (Degree = 2 & 3)
mse_poly = {}
r2_poly = {}

for degree in [2, 3]:
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(X_train, Y_train)
    
    Y_pred_poly = poly_model.predict(X_test)
    mse_poly[degree] = mean_squared_error(Y_test, Y_pred_poly)
    r2_poly[degree] = r2_score(Y_test, Y_pred_poly)

# 6️⃣ KNN Regression untuk K = 3, 5, 7
k_values = [3, 5, 7]
results = {}

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, Y_train)
    
    Y_pred_knn = knn.predict(X_test)
    
    mse_knn = mean_squared_error(Y_test, Y_pred_knn)
    r2_knn = r2_score(Y_test, Y_pred_knn)
    
    results[k] = {"MSE": mse_knn, "R²": r2_knn}

# 7️⃣ Tampilkan Perbandingan Performa Model
print("\n📌 **Perbandingan Performa Model:**")
print(f"{'Model':<30} {'MSE':<15} {'R²':<10}")
print("-" * 55)
print(f"{'Linear Regression':<30} {mse_linear:.2f} {r2_linear:.4f}")

for degree in [2, 3]:
    print(f"{'Polynomial Regression (deg='+str(degree)+')':<30} {mse_poly[degree]:.2f} {r2_poly[degree]:.4f}")

for k in k_values:
    print(f"{'KNN Regression (K='+str(k)+')':<30} {results[k]['MSE']:.2f} {results[k]['R²']:.4f}")
