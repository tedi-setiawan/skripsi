# Analisis Prediksi Harga Beras di Pasar Tradisional Sumedang

> **Skripsi – Program Studi S1 Teknik Informatika**
> **Penulis:** **Tedi Setiawan** (NPM 2113218028)
> **Judul:** *Analisis Prediksi Harga Beras di Pasar Tradisional Sumedang Menggunakan Metode Regresi Linear*

---

## 📜 Latar Belakang

Pasar‑pasar tradisional di Kabupaten Sumedang (Tanjungsari, Conggeang, Parakanmuncang, Darmaraja) menjadi pusat distribusi utama beras bagi masyarakat lokal. Fluktuasi harga yang tajam menyulitkan:

* **Konsumen** – sulit merencanakan pengeluaran pangan.
* **Petani & pedagang** – pendapatan tergerus akibat ketidakpastian harga.

Penelitian ini memanfaatkan **metode regresi linear** untuk memodelkan hubungan harga *Beras Premium* (variabel independen) terhadap harga *Beras Medium* (variabel dependen) sehingga pemangku kepentingan mendapatkan **prediksi cepat** harga yang lebih stabil.

## 🎯 Tujuan Penelitian

1. **Membangun model regresi linear** untuk memprediksi harga *Beras Medium* di pasar tradisional Sumedang.
2. **Menyediakan antarmuka web** berbasis Streamlit agar pengguna dapat:

   * Melakukan prediksi manual.
   * Melakukan batch‑prediction melalui unggah CSV.
3. Mendemonstrasikan manfaat data science dalam pengambilan keputusan harga pangan lokal.

## 🗂️ Batasan Masalah

| No | Batasan                                                                                            |
| -- | -------------------------------------------------------------------------------------------------- |
| 1  | Fokus pada empat pasar tradisional Sumedang: **Tanjungsari, Conggeang, Parakanmuncang, Darmaraja** |
| 2  | Komoditas yang dianalisis: **Beras Premium** dan **Beras Medium**                                  |
| 3  | Periode data: **2022 – 2024**                                                                      |
| 4  | Sumber data: **Dinas Koperasi UKM Perdagangan & Perindustrian Kab. Sumedang**                      |

## 🏗️ Arsitektur Proyek

```
skripsi/
├── app.py                                 # Aplikasi Streamlit
├── model_regresi_beras.pkl                # Model linier ter‑training
├── requirements.txt                       # Dependensi Python
├── contoh_dataset_prediksi.csv            # (opsional) Dataset mentah/olah
└── README.md                              # Dokumentasi ini
```

## ⚙️ Teknologi & Library

* **Python ≥ 3.9**
* [Streamlit](https://streamlit.io/)
* Pandas, NumPy, Scikit‑learn, Joblib
* Matplotlib & Seaborn (visualisasi)

## 🚀 Cara Menjalankan Lokal

```bash
# 1 – Klon repositori
git clone https://github.com/tedi-setiawan/skripsi.git
cd skripsi

# 2 – Buat lingkungan virtual (opsional)
python -m venv venv
source venv/Scripts/activate  # Windows

# 3 – Instal dependensi
pip install -r requirements.txt

# 4 – Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka di [http://localhost:8501](http://localhost:8501).

## 🌐 Deployment ke Streamlit Cloud

1. Fork/clone repo ini ke akun GitHub Anda.
2. Masuk ke **Streamlit Community Cloud** → **New app**.
3. Pilih repository & branch `main`, file `app.py`.
4. Klik **Deploy**.

## 📈 Cara Menggunakan

### 1. Halaman **Metodologi** (default)

Menjelaskan tujuan, metode, dan spesifikasi data.

### 2. Halaman **EDA & Upload CSV**

* Unggah file `CSV` dengan kolom `tanggal, nama_pasar, Beras Premium, Beras Medium` (kosong).
* Aplikasi:

  * Membersihkan data
  * Menampilkan statistik & plot
  * Mengisi kolom **Beras Medium** hasil prediksi
  * Menyediakan download CSV hasil

### 3. Halaman **Prediksi Manual**

* Masukkan harga **Beras Premium** → aplikasi menampilkan prediksi harga **Beras Medium** secara real‑time.

## 📊 Hasil Utama (Ringkasan)

| Metrik | Nilai          |
| ------ | -------------- |
| R²     | 0.948          |
| MAE    | Rp 236.91      |
| RMSE   | Rp 326.05      |
| MAPE   | 2.00 %         |

> \*Nilai bervariasi tergantung data training terkini.

## 🔮 Potensi Pengembangan

* Penambahan fitur ekternal (volume panen, cuaca, inflasi).
* Model non‑linear (Random Forest, XGBoost).
* Forecasting multi‑step (ARIMA/LSTM).

## 📝 Lisensi

Repositori ini menggunakan lisensi **MIT** – silakan gunakan dan modifikasi dengan tetap mencantumkan atribusi.

## 🙏 Acknowledgements

Data disediakan oleh **Dinas Koperasi, UKM, Perdagangan & Perindustrian Kabupaten Sumedang**.
Terima kasih kepada dosen pembimbing, rekan‑rekan, dan keluarga yang telah mendukung penelitian ini.
