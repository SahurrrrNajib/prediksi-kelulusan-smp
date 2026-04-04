from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# =========================
# LOAD & TRAIN MODEL
# =========================
df = pd.read_csv("jumlah_lulus.csv")

X = df[['tahun']]
Y = df['jumlah siswa lulus']

model = LinearRegression()
model.fit(X, Y)

# =========================
# ROUTE UTAMA
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediksi = None
    tahun_input = None

    if request.method == "POST":
        tahun_input = int(request.form["tahun"])

        # prediksi
        prediksi = model.predict([[tahun_input]])[0]

        # =========================
        # BUAT GRAFIK
        # =========================
        plt.figure()

        plt.scatter(X, Y, label="Data Asli")
        plt.plot(X, model.predict(X), color='red', label="Regresi")

        # titik prediksi
        plt.scatter(tahun_input, prediksi, color='green', label="Prediksi")

        plt.xlabel("Tahun")
        plt.ylabel("Jumlah Lulus")
        plt.legend()

        plt.savefig("static/plot.png")
        plt.close()

    return render_template("index.html", prediksi=prediksi, tahun=tahun_input)

if __name__ == "__main__":
    app.run(debug=True)
