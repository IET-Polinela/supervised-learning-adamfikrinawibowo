import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset tanpa outlier
file_path = "house_pricing_no_outliers.csv"  # Gunakan dataset tanpa outlier
df_no_outliers = pd.read_csv(file_path)

# Identifikasi fitur numerik
numeric_features = df_no_outliers.select_dtypes(include=[np.number]).columns  # Hanya fitur numerik

# 1️⃣ Menerapkan StandardScaler
scaler_standard = StandardScaler()
df_scaled_standard = pd.DataFrame(scaler_standard.fit_transform(df_no_outliers[numeric_features]),
                                  columns=numeric_features)

# 2️⃣ Menerapkan MinMaxScaler
scaler_minmax = MinMaxScaler()
df_scaled_minmax = pd.DataFrame(scaler_minmax.fit_transform(df_no_outliers[numeric_features]),
                                columns=numeric_features)

# 3️⃣ Menyimpan dataset hasil scaling
df_scaled_standard.to_csv("house_pricing_standard_scaled.csv", index=False)
df_scaled_minmax.to_csv("house_pricing_minmax_scaled.csv", index=False)

print("\n📂 Dataset hasil scaling telah disimpan:")
print("- 'house_pricing_standard_scaled.csv' (StandardScaler)")
print("- 'house_pricing_minmax_scaled.csv' (MinMaxScaler)")

# 4️⃣ Visualisasi distribusi sebelum dan sesudah scaling
plt.figure(figsize=(15, 5))

# Sebelum scaling
plt.subplot(1, 3, 1)
df_no_outliers[numeric_features].hist(bins=30, figsize=(15, 5), edgecolor='black', alpha=0.7)
plt.title("📌 Distribusi Data Sebelum Scaling")

# Setelah StandardScaler
plt.subplot(1, 3, 2)
df_scaled_standard.hist(bins=30, figsize=(15, 5), edgecolor='black', alpha=0.7)
plt.title("📌 Distribusi Data Setelah StandardScaler")

# Setelah MinMaxScaler
plt.subplot(1, 3, 3)
df_scaled_minmax.hist(bins=30, figsize=(15, 5), edgecolor='black', alpha=0.7)
plt.title("📌 Distribusi Data Setelah MinMaxScaler")

plt.tight_layout()
plt.show()
