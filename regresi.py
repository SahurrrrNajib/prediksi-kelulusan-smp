import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("jumlah_lulus.csv")

print("=== DATA AWAL ===")
print(df)

# =========================
# 2. BERSIHKAN DATA
# =========================
df = df.dropna()
df = df.sort_values(by='tahun')

# =========================
# 3. VISUALISASI DATA
# =========================
plt.figure(figsize=(8,5))
plt.plot(df['tahun'], df['jumlah siswa lulus'], marker='o')

plt.title("Jumlah Kelulusan SMP")
plt.xlabel("Tahun")
plt.ylabel("Jumlah Lulus")
plt.show()

# =========================
# 4. SIAPKAN DATA
# =========================
X = df[['tahun']]
Y = df['jumlah siswa lulus']

# =========================
# 5. TRAIN MODEL
# =========================
model = LinearRegression()
model.fit(X, Y)

# =========================
# 6. PREDIKSI DATA LATIH
# =========================
Y_pred = model.predict(X)

# =========================
# 7. EVALUASI
# =========================
mae = mean_absolute_error(Y, Y_pred)
mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

print("\n=== HASIL EVALUASI ===")
print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)

# =========================
# 8. VISUALISASI REGRESI
# =========================
plt.figure(figsize=(8,5))
plt.scatter(X, Y, label="Data Aktual")
plt.plot(X, Y_pred, color='red', label="Regresi")

plt.xlabel("Tahun")
plt.ylabel("Jumlah Lulus")
plt.title("Regresi Linear Kelulusan")
plt.legend()
plt.show()

# =========================
# 9. PREDIKSI MASA DEPAN
# =========================
tahun_prediksi = np.array([[2024], [2025], [2026]])
hasil_prediksi = model.predict(tahun_prediksi)

print("\n=== PREDIKSI MASA DEPAN ===")
for i, val in enumerate(hasil_prediksi):
    print(f"Tahun {tahun_prediksi[i][0]}: {int(val)} siswa")
