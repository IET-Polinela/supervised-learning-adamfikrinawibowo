import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "/content/house_pricing.csv"  # Sesuaikan dengan lokasi file
df = pd.read_csv(file_path)

# 1️⃣ Encoding fitur kategorikal menggunakan One-Hot Encoding
df_encoded = pd.get_dummies(df, drop_first=True)  # Menghindari dummy trap

# 2️⃣ Memisahkan fitur independent (X) dan target (Y)
X = df_encoded.drop(columns=["SalePrice"])  # Semua fitur kecuali target
Y = df_encoded["SalePrice"]  # Target

# 3️⃣ Membagi dataset menjadi training (80%) dan testing (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4️⃣ Menampilkan hasil setelah preprocessing
print("\n📌 **Hasil Data Preprocessing:**")
print(f"Total fitur setelah encoding: {X.shape[1]}")
print(f"Jumlah data training: {X_train.shape[0]} sampel, dengan {X_train.shape[1]} fitur")
print(f"Jumlah data testing: {X_test.shape[0]} sampel, dengan {X_test.shape[1]} fitur")

# Simpan dataset hasil preprocessing jika diperlukan
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
Y_train.to_csv("Y_train.csv", index=False)
Y_test.to_csv("Y_test.csv", index=False)

print("\n📂 Dataset hasil preprocessing telah disimpan sebagai 'X_train.csv', 'X_test.csv', 'Y_train.csv', dan 'Y_test.csv'.")
