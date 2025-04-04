import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# 1️⃣ Load dataset yang sudah di-scaling
file_path = "house_pricing_standard_scaled.csv"
df_scaled = pd.read_csv(file_path)

# Pisahkan fitur (X) dan target (Y)
X = df_scaled.drop(columns=["SalePrice"], errors="ignore")  # Hapus target dari X
Y = df_scaled["SalePrice"] if "SalePrice" in df_scaled.columns else None  # Ambil target

# Cek jika Y masih None (berarti dataset tidak punya target)
if Y is None:
    raise ValueError("Kolom 'SalePrice' tidak ditemukan dalam dataset!")

# Split data menjadi Training (80%) dan Testing (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 2️⃣ Hapus kolom yang 100% kosong sebelum imputasi
X_train = X_train.dropna(axis=1, how="all")
X_test = X_test.dropna(axis=1, how="all")

# Pastikan jumlah kolom di train & test tetap sama setelah penghapusan
common_cols = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# 3️⃣ Mengatasi missing values dengan SimpleImputer
imputer = SimpleImputer(strategy="mean")  # Isi NaN dengan mean tiap kolom
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=common_cols)
X_test = pd.DataFrame(imputer.transform(X_test), columns=common_cols)

# Pastikan tidak ada missing values setelah imputasi
print("\n✅ Missing values setelah imputasi:")
print(f"X_train: {X_train.isnull().sum().sum()} missing values")
print(f"X_test: {X_test.isnull().sum().sum()} missing values")

# 4️⃣ Melatih model Linear Regression
model = LinearRegression()
model.fit(X_train, Y_train)

# 5️⃣ Prediksi pada training dan testing set
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# 6️⃣ Evaluasi model
mse_train = mean_squared_error(Y_train, Y_train_pred)
r2_train = r2_score(Y_train, Y_train_pred)

mse_test = mean_squared_error(Y_test, Y_test_pred)
r2_test = r2_score(Y_test, Y_test_pred)

# 7️⃣ Menampilkan hasil evaluasi
print("\n📌 **Evaluasi Model Linear Regression:**")
print(f"Training Set: MSE = {mse_train:.2f}, R² = {r2_train:.4f}")
print(f"Testing Set: MSE = {mse_test:.2f}, R² = {r2_test:.4f}")

# 8️⃣ Visualisasi Hasil Prediksi vs Nilai Aktual
plt.figure(figsize=(12, 5))

# 📌 Scatter plot Prediksi vs Aktual
plt.subplot(1, 2, 1)
plt.scatter(Y_test, Y_test_pred, alpha=0.5, color="blue")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "--", color="red")
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("📌 Scatter Plot: Prediksi vs Aktual")
plt.savefig("scatter_plot_linear_regression.png", bbox_inches="tight")  # Simpan sebagai PNG

# 📌 Residual Plot
plt.subplot(1, 2, 2)
residuals = Y_test - Y_test_pred
plt.scatter(Y_test_pred, residuals, alpha=0.5, color="green")
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Sale Price")
plt.ylabel("Residuals")
plt.title("📌 Residual Plot")
plt.savefig("residual_plot_linear_regression.png", bbox_inches="tight")  # Simpan sebagai PNG

plt.tight_layout()
plt.show()

print("\n📂 Visualisasi telah disimpan sebagai:")
print("- 'scatter_plot_linear_regression.png'")
print("- 'residual_plot_linear_regression.png'")
