import pandas as pd
from tabulate import tabulate

# Load dataset
file_path = "/content/house_pricing.csv"  # Sesuaikan dengan lokasi file
df = pd.read_csv(file_path)

# Menampilkan informasi dataset
print("\n📌 **Informasi Dataset:**")
df_info = df.info()

# Menampilkan statistik deskriptif hanya untuk fitur numerik
desc_stats = df.describe().T  # Transpose agar lebih mudah dibaca
print("\n📌 **Statistik Deskriptif:**")
print(tabulate(desc_stats, headers="keys", tablefmt="pretty"))

# Mengecek missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

# Menampilkan missing values dalam bentuk tabel
if not missing_values.empty:
    print("\n📌 **Missing Values:**")
    missing_df = pd.DataFrame({"Kolom": missing_values.index, "Jumlah Missing": missing_values.values})
    print(tabulate(missing_df, headers="keys", tablefmt="pretty"))
else:
    print("\n✅ Tidak ada missing values dalam dataset.")

# Rekomendasi handling missing values
columns_to_drop = ["PoolQC", "Fence", "MiscFeature", "Alley"]  # Kolom dengan missing > 90%
df_cleaned = df.drop(columns=columns_to_drop)

# Mengisi nilai yang hilang berdasarkan kategori yang sesuai
df_cleaned["LotFrontage"].fillna(df_cleaned.groupby("Neighborhood")["LotFrontage"].transform("median"), inplace=True)
df_cleaned["MasVnrType"].fillna("None", inplace=True)
df_cleaned["MasVnrArea"].fillna(0, inplace=True)
df_cleaned["Electrical"].fillna(df_cleaned["Electrical"].mode()[0], inplace=True)

# Mengisi atribut Basement dan Garage dengan "None" untuk kategori dan 0 untuk numerik
basement_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
garage_cols = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]

for col in basement_cols + garage_cols:
    df_cleaned[col].fillna("None", inplace=True)

df_cleaned["GarageYrBlt"].fillna(0, inplace=True)

# Cek missing values setelah ditangani
remaining_missing = df_cleaned.isnull().sum().sum()
if remaining_missing == 0:
    print("\n✅ Semua missing values sudah ditangani.")
else:
    print(f"\n⚠️ Masih ada {remaining_missing} missing values yang perlu dicek.")

# Simpan hasil cleaning jika diperlukan
df_cleaned.to_csv("house_pricing_cleaned.csv", index=False)
print("\n📂 Dataset yang sudah dibersihkan disimpan sebagai 'house_pricing_cleaned.csv'.")
