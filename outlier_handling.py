import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
file_path = "/content/house_pricing.csv"  # Sesuaikan dengan lokasi file
df = pd.read_csv(file_path)

# 1️⃣ Identifikasi fitur numerik
numeric_features = df.select_dtypes(include=[np.number]).columns  # Hanya fitur numerik

# 2️⃣ Fungsi untuk menghapus outlier menggunakan metode IQR
def remove_outliers_iqr(data, threshold=1.5):
    """Menghapus outlier menggunakan metode Interquartile Range (IQR)."""
    df_clean = data.copy()
    for col in numeric_features:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# 3️⃣ Buat dataset tanpa outlier
df_no_outliers = remove_outliers_iqr(df)

# 4️⃣ Simpan dataset:
df.to_csv("house_pricing_with_outliers.csv", index=False)  # Dataset asli (dengan outlier)
df_no_outliers.to_csv("house_pricing_no_outliers.csv", index=False)  # Dataset tanpa outlier

# 5️⃣ Menampilkan perbandingan jumlah data sebelum & sesudah outlier handling
print("\n📌 **Perbandingan Data Sebelum & Sesudah Outlier Handling:**")
print(f"Jumlah data awal: {df.shape[0]} sampel")
print(f"Jumlah data setelah menghapus outlier: {df_no_outliers.shape[0]} sampel")

# 6️⃣ Visualisasi Boxplot (DIPISAH dan DISIMPAN)

# 📌 Boxplot untuk dataset dengan outlier
plt.figure(figsize=(10, 6))
df[numeric_features].boxplot(rot=90)
plt.title("📌 Boxplot Dataset dengan Outlier")
plt.xticks(rotation=90)
plt.savefig("boxplot_with_outliers.png", bbox_inches="tight")  # Simpan sebagai PNG
plt.show()

# 📌 Boxplot untuk dataset tanpa outlier
plt.figure(figsize=(10, 6))
df_no_outliers[numeric_features].boxplot(rot=90)
plt.title("📌 Boxplot Dataset tanpa Outlier")
plt.xticks(rotation=90)
plt.savefig("boxplot_no_outliers.png", bbox_inches="tight")  # Simpan sebagai PNG
plt.show()

print("\n📂 Visualisasi telah disimpan sebagai 'boxplot_with_outliers.png' dan 'boxplot_no_outliers.png'.")
