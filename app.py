# app.py
# -------------------------------------------------------------
# Streamlit: Prediksi Harga Beras Medium di Pasar Tradisional
# -------------------------------------------------------------

# ➊  ── HARUS PALING ATAS ──
import streamlit as st

st.set_page_config(
    page_title="Prediksi Beras Medium",
    page_icon="🌾",
    layout="wide"
)

# -------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)

# -------------------------------------------------------------
# Load model with error handling
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        if not os.path.exists("model_regresi_beras.pkl"):
            st.error("❌ File model 'model_regresi_beras.pkl' tidak ditemukan!")
            st.info("Pastikan file model sudah ada di direktori yang sama dengan app.py")
            return None
        return joblib.load("model_regresi_beras.pkl")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_model()

# -------------------------------------------------------------
# App Header
# -------------------------------------------------------------
st.title("🌾 Prediksi Harga Beras Medium")

st.markdown("""
**Judul Skripsi**&emsp;&nbsp;: *Analisis Prediksi Harga Beras di Pasar Tradisional Sumedang Menggunakan Metode Regresi Linear*  
**Nama**&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Tedi Setiawan  
**NPM**&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;: 2113218028
""")


# Only proceed if model is loaded successfully
if model is not None:
    # -------------------------------------------------------------
    # Sidebar Navigation
    # -------------------------------------------------------------
    st.sidebar.title("🌾 Menu Navigasi")
    page = st.sidebar.radio(
        "Pilih Halaman:", 
        ["📖 Metodologi", "📊 EDA & Upload CSV", "🔮 Prediksi Manual"]
    )


    # -------------------------------------------------------------
    # 1) Halaman EDA & Upload CSV
    # -------------------------------------------------------------
    if page == "📊 EDA & Upload CSV":
        st.header("📊 Eksplorasi Data & Prediksi via CSV")

        # ----- Contoh format dataset -----
        st.subheader("📋 Format CSV yang Diperlukan")
        example_df = pd.DataFrame({
            "tanggal": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "nama_pasar": ["Pasar Darmaraja", "Pasar Darmaraja", "Pasar Darmaraja", "Pasar Parakanmuncang", "Pasar Parakanmuncang"],
            "Beras Premium": [12496, 13237, 12408, 12400, 12306],
            "Beras Medium": ["", "", "", "", ""]
        })
        st.dataframe(example_df, use_container_width=True)

        st.markdown("""
        **📌 Petunjuk:**
        * File CSV harus memiliki kolom: `tanggal`, `nama_pasar`, `Beras Premium`, `Beras Medium`
        * Kolom `tanggal` dalam format YYYY-MM-DD (contoh: 2024-01-01)
        * Kolom `Beras Premium` berisi harga dalam rupiah (tanpa titik/koma)
        * Kolom `Beras Medium` boleh kosong (akan diisi prediksi)
        * Contoh harga: `12500` untuk Rp 12.500
        """)

        # ----- Upload file untuk prediksi -----
        st.subheader("📁 Upload File CSV")
        uploaded = st.file_uploader(
            "Pilih file CSV untuk prediksi harga Beras Medium", 
            type="csv"
        )

        if uploaded:
            try:
                df_input = pd.read_csv(uploaded)
                
                # Validate data
                if df_input.empty:
                    st.error("❌ File CSV kosong!")
                else:
                    # Check required columns
                    required_cols = ["tanggal", "nama_pasar", "Beras Premium"]
                    missing_cols = [col for col in required_cols if col not in df_input.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Kolom yang hilang: {', '.join(missing_cols)}")
                        st.info("Pastikan CSV memiliki kolom: tanggal, nama_pasar, Beras Premium, Beras Medium")
                    else:
                        # Clean and validate data
                        df_clean = df_input.copy()
                        
                        # Convert tanggal to datetime
                        try:
                            df_clean['tanggal'] = pd.to_datetime(df_clean['tanggal'])
                            df_clean = df_clean.sort_values('tanggal')
                        except:
                            st.error("❌ Format tanggal tidak valid! Gunakan format YYYY-MM-DD")
                            st.stop()
                        
                        # Validate Beras Premium data
                        if not pd.api.types.is_numeric_dtype(df_clean["Beras Premium"]):
                            # Try to convert to numeric, removing any non-numeric characters
                            df_clean["Beras Premium"] = pd.to_numeric(df_clean["Beras Premium"], errors='coerce')
                        
                        # Remove rows with invalid Beras Premium data
                        df_clean = df_clean.dropna(subset=["Beras Premium"])
                        
                        if df_clean.empty:
                            st.error("❌ Tidak ada data valid untuk diprediksi!")
                        else:
                            # === EDA SECTION ===
                            st.subheader("📊 Exploratory Data Analysis")
                            
                            # Show data info
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Data", len(df_clean))
                            with col2:
                                st.metric("Jumlah Pasar", df_clean['nama_pasar'].nunique())
                            with col3:
                                period_days = (df_clean['tanggal'].max() - df_clean['tanggal'].min()).days
                                st.metric("Periode (Hari)", period_days)
                            
                            # Time series plot for Beras Premium - Separate plots for each market
                            st.subheader("📈 Time Series Harga Beras Premium per Pasar")
                            
                            markets = df_clean['nama_pasar'].unique()
                            colors = plt.cm.Set3(np.linspace(0, 1, len(markets)))
                            
                            # Create subplot layout
                            n_markets = len(markets)
                            cols = 2 if n_markets > 1 else 1
                            rows = (n_markets + cols - 1) // cols
                            
                            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
                            if n_markets == 1:
                                axes = [axes]
                            elif rows == 1:
                                axes = axes.reshape(1, -1)
                            
                            for i, (market, color) in enumerate(zip(markets, colors)):
                                row = i // cols
                                col = i % cols
                                ax = axes[row, col] if rows > 1 else axes[col]
                                
                                market_data = df_clean[df_clean['nama_pasar'] == market]
                                ax.plot(market_data['tanggal'], 
                                       market_data['Beras Premium'], 
                                       marker='o', 
                                       color=color,
                                       linewidth=2,
                                       markersize=6)
                                
                                ax.set_title(f"{market}", fontsize=12, fontweight='bold')
                                ax.set_xlabel("Tanggal")
                                ax.set_ylabel("Harga Beras Premium (Rp)")
                                ax.grid(True, alpha=0.3)
                                ax.tick_params(axis='x', rotation=45)
                                
                                # Format y-axis to show currency
                                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp {x:,.0f}'))
                            
                            # Hide empty subplots
                            for i in range(n_markets, rows * cols):
                                row = i // cols
                                col = i % cols
                                if rows > 1:
                                    axes[row, col].set_visible(False)
                                elif cols > 1:
                                    axes[col].set_visible(False)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # === PREDICTION SECTION ===
                            st.subheader("🔮 Melakukan Prediksi...")
                            
                            # Make predictions
                            try:
                                y_pred = model.predict(df_clean[["Beras Premium"]])
                                df_result = df_clean.copy()
                                df_result["Beras Medium"] = y_pred.round(0).astype(int)

                                st.success("✅ Berhasil memprediksi harga beras medium!")
                                
                                # Display results
                                st.subheader("📈 Hasil Prediksi")
                                
                                # Format the display dataframe
                                display_df = df_result.copy()
                                display_df['tanggal'] = display_df['tanggal'].dt.strftime('%Y-%m-%d')
                                display_df['Beras Premium'] = display_df['Beras Premium'].apply(lambda x: f"Rp {x:,.0f}")
                                display_df['Beras Medium'] = display_df['Beras Medium'].apply(lambda x: f"Rp {x:,.0f}")
                                
                                st.dataframe(display_df, use_container_width=True)

                                # Time series plot for predicted Beras Medium - Separate plots for each market
                                st.subheader("📈 Time Series Hasil Prediksi Beras Medium per Pasar")
                                
                                # Create subplot layout for Medium predictions
                                fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
                                if n_markets == 1:
                                    axes = [axes]
                                elif rows == 1:
                                    axes = axes.reshape(1, -1)
                                
                                for i, (market, color) in enumerate(zip(markets, colors)):
                                    row = i // cols
                                    col = i % cols
                                    ax = axes[row, col] if rows > 1 else axes[col]
                                    
                                    market_data = df_result[df_result['nama_pasar'] == market]
                                    ax.plot(market_data['tanggal'], 
                                           market_data['Beras Medium'], 
                                           marker='s', 
                                           color=color,
                                           linewidth=2,
                                           markersize=6,
                                           linestyle='--')
                                    
                                    ax.set_title(f"{market} - Prediksi Medium", fontsize=12, fontweight='bold')
                                    ax.set_xlabel("Tanggal")
                                    ax.set_ylabel("Harga Beras Medium (Rp)")
                                    ax.grid(True, alpha=0.3)
                                    ax.tick_params(axis='x', rotation=45)
                                    
                                    # Format y-axis to show currency
                                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp {x:,.0f}'))
                                
                                # Hide empty subplots
                                for i in range(n_markets, rows * cols):
                                    row = i // cols
                                    col = i % cols
                                    if rows > 1:
                                        axes[row, col].set_visible(False)
                                    elif cols > 1:
                                        axes[col].set_visible(False)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Comparison plot - Separate plots for each market
                                st.subheader("📊 Perbandingan Harga Premium vs Medium per Pasar")
                                
                                # Create subplot layout for comparison
                                fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
                                if n_markets == 1:
                                    axes = [axes]
                                elif rows == 1:
                                    axes = axes.reshape(1, -1)
                                
                                for i, (market, color) in enumerate(zip(markets, colors)):
                                    row = i // cols
                                    col = i % cols
                                    ax = axes[row, col] if rows > 1 else axes[col]
                                    
                                    market_data = df_result[df_result['nama_pasar'] == market]
                                    
                                    # Plot Premium
                                    ax.plot(market_data['tanggal'], 
                                           market_data['Beras Premium'], 
                                           marker='o', 
                                           label='Premium', 
                                           color=color,
                                           linewidth=2,
                                           markersize=5)
                                    
                                    # Plot Medium
                                    ax.plot(market_data['tanggal'], 
                                           market_data['Beras Medium'], 
                                           marker='s', 
                                           label='Medium (Prediksi)', 
                                           color=color,
                                           linewidth=2,
                                           markersize=5,
                                           linestyle='--',
                                           alpha=0.8)
                                    
                                    ax.set_title(f"{market} - Premium vs Medium", fontsize=12, fontweight='bold')
                                    ax.set_xlabel("Tanggal")
                                    ax.set_ylabel("Harga (Rp)")
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                    ax.tick_params(axis='x', rotation=45)
                                    
                                    # Format y-axis to show currency
                                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp {x:,.0f}'))
                                
                                # Hide empty subplots
                                for i in range(n_markets, rows * cols):
                                    row = i // cols
                                    col = i % cols
                                    if rows > 1:
                                        axes[row, col].set_visible(False)
                                    elif cols > 1:
                                        axes[col].set_visible(False)
                                
                                plt.tight_layout()
                                st.pyplot(fig)

                                # Summary statistics
                                st.subheader("📊 Statistik Ringkas")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Beras Premium:**")
                                    premium_stats = df_result.groupby('nama_pasar')['Beras Premium'].agg(['mean', 'min', 'max', 'std']).round(0)
                                    premium_stats.columns = ['Rata-rata', 'Minimum', 'Maksimum', 'Std Deviasi']
                                    for col in premium_stats.columns:
                                        premium_stats[col] = premium_stats[col].apply(lambda x: f"Rp {x:,.0f}")
                                    st.dataframe(premium_stats)
                                
                                with col2:
                                    st.markdown("**Beras Medium (Prediksi):**")
                                    medium_stats = df_result.groupby('nama_pasar')['Beras Medium'].agg(['mean', 'min', 'max', 'std']).round(0)
                                    medium_stats.columns = ['Rata-rata', 'Minimum', 'Maksimum', 'Std Deviasi']
                                    for col in medium_stats.columns:
                                        medium_stats[col] = medium_stats[col].apply(lambda x: f"Rp {x:,.0f}")
                                    st.dataframe(medium_stats)

                                # Download results in original format
                                st.subheader("💾 Download Hasil")
                                
                                # Prepare CSV in original format (same as uploaded)
                                download_df = df_result.copy()
                                download_df['tanggal'] = download_df['tanggal'].dt.strftime('%Y-%m-%d')
                                download_df = download_df[['tanggal', 'nama_pasar', 'Beras Premium', 'Beras Medium']]
                                
                                csv = download_df.to_csv(index=False)
                                st.download_button(
                                    label="💾 Download Hasil Prediksi (CSV)",
                                    data=csv,
                                    file_name=f"hasil_prediksi_beras_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                
                                st.info("📁 File hasil prediksi memiliki format yang sama dengan file upload, dengan kolom 'Beras Medium' yang sudah terisi prediksi.")
                                        
                            except Exception as e:
                                st.error(f"❌ Gagal melakukan prediksi: {e}")
                                
            except Exception as e:
                st.error(f"❌ Gagal membaca file CSV: {e}")
                st.info("Pastikan file adalah CSV yang valid dengan encoding UTF-8.")

    # -------------------------------------------------------------
    # 2) Halaman Prediksi Manual
    # -------------------------------------------------------------
    elif page == "🔮 Prediksi Manual":
        st.header("🔮 Prediksi Manual Harga Beras Medium")

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("💰 Input Harga")
            harga_premium = st.number_input(
                "Masukkan Harga Beras Premium (Rp)",
                min_value=5000,
                max_value=100000,
                step=100,
                value=13000,
                help="Masukkan harga beras premium dalam rupiah"
            )
            
            # Real-time prediction
            if harga_premium > 0:
                try:
                    pred = model.predict(np.array([[harga_premium]]))[0]
                    
                    st.markdown("### 🎯 Hasil Prediksi")
                    st.success(f"**Perkiraan Harga Beras Medium: Rp {int(pred):,}**")
                    
                    # Calculate percentage difference
                    diff_pct = ((pred - harga_premium) / harga_premium) * 100
                    if diff_pct > 0:
                        st.info(f"📈 Beras medium lebih mahal {diff_pct:.1f}% dari premium")
                    else:
                        st.info(f"📉 Beras medium lebih murah {abs(diff_pct):.1f}% dari premium")
                        
                except Exception as e:
                    st.error(f"❌ Gagal melakukan prediksi: {e}")
        
        with col2:
            st.subheader("💡 Tips & Informasi")
            st.markdown("""
            **Cara Penggunaan:**
            * Masukkan harga beras premium yang ingin diprediksi
            * Hasil prediksi akan muncul secara otomatis
            * Gunakan slider atau ketik langsung untuk input yang cepat
            
            **Catatan Penting:**
            * Model ini menggunakan regresi linear sederhana
            * Akurasi bergantung pada kesesuaian dengan data training
            * Harga aktual dapat berbeda dengan prediksi
            * Gunakan sebagai referensi, bukan patokan mutlak
            """)
            
            # Sample predictions table
            st.subheader("📋 Contoh Prediksi")
            sample_prices = [10000, 12000, 14000, 16000, 18000]
            sample_preds = []
            
            for price in sample_prices:
                try:
                    pred = model.predict(np.array([[price]]))[0]
                    sample_preds.append(int(pred))
                except:
                    sample_preds.append("Error")
            
            sample_df = pd.DataFrame({
                "Premium (Rp)": [f"{p:,}" for p in sample_prices],
                "Prediksi Medium (Rp)": [f"{p:,}" if isinstance(p, int) else p for p in sample_preds]
            })
            st.dataframe(sample_df, use_container_width=True)

    # -------------------------------------------------------------
    # 3) Halaman Metodologi
    # -------------------------------------------------------------
    else:
        st.header("📖 Metodologi & Penjelasan Model")
        
        # Overview
        st.subheader("🎯 Tujuan Penelitian")
        st.markdown("""
        Aplikasi ini bertujuan untuk memprediksi harga beras medium berdasarkan harga beras premium 
        di pasar tradisional. Prediksi ini dapat membantu:
        
        * **Pedagang** dalam menentukan strategi harga
        * **Konsumen** dalam merencanakan pembelian
        * **Peneliti** dalam menganalisis dinamika harga komoditas
        * **Pemerintah** dalam monitoring stabilitas pangan
        """)
        
        # Methodology
        st.subheader("🔬 Metode yang Digunakan")
        
        # Model Description
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            #### 1. Model Regresi Linear
            
            **Formula Matematis:**
            ```
            Harga Beras Medium = β₀ + β₁ × Harga Beras Premium + ε
            ```
            
            Dimana:
            * `β₀` (intercept) = konstanta dasar
            * `β₁` (slope) = koefisien hubungan linear
            * `ε` (error) = residual/kesalahan prediksi
            
            **Asumsi Model:**
            * Hubungan linear antara harga premium dan medium
            * Residual terdistribusi normal
            * Homoskedastisitas (varians konstan)
            * Tidak ada multikolinearitas (untuk model multivariat)
            """)
        
        with col2:
            st.info("""
            **📊 Keunggulan:**
            ✅ Mudah diinterpretasi
            ✅ Cepat dalam prediksi
            ✅ Tidak memerlukan data besar
            ✅ Transparan dan explainable
            
            **⚠️ Keterbatasan:**
            ❌ Asumsi hubungan linear
            ❌ Sensitif terhadap outlier
            ❌ Tidak menangkap pola non-linear
            """)
        
        # Data Requirements
        st.subheader("📊 Spesifikasi Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Input Data yang Diperlukan:
            
            **📁 Format CSV:**
            * `tanggal`: Format YYYY-MM-DD
            * `nama_pasar`: Nama lokasi pasar
            * `Beras Premium`: Harga dalam rupiah (numerik)
            * `Beras Medium`: Kosong (akan diprediksi)
            
            **🔍 Preprocessing:**
            * Validasi format tanggal
            * Konversi data numerik
            * Penanganan missing values
            * Sorting berdasarkan kronologi
            """)
        
        with col2:
            st.markdown("""
            #### Asumsi Data:
            
            **📈 Karakteristik:**
            * Data time series dari pasar tradisional
            * Harga dalam mata uang rupiah
            * Konsistensi unit pengukuran
            * Tidak ada outlier ekstrem
            
            **🎯 Target:**
            * Prediksi harga beras medium
            * Akurasi tinggi untuk rentang normal
            * Generalisasi untuk pasar serupa
            """)
        
        # Model Performance
        st.subheader("📈 Evaluasi Model")
        
        # Create sample metrics (placeholder - in real app these would come from actual model evaluation)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", "0.948", "📊")
        with col2:
            st.metric("MAE", "Rp 236.91", "📉")
        with col3:
            st.metric("RMSE", "Rp 326.05", "📊")
        with col4:
            st.metric("MAPE", "2.00%", "📈")
        
        st.markdown("""
        **📋 Interpretasi Metrik:**
        
        * **R² Score (0.85)**: Model menjelaskan 85% variasi dalam data
        * **MAE (Mean Absolute Error)**: Rata-rata kesalahan absolut Rp 245
        * **RMSE (Root Mean Square Error)**: Standar deviasi kesalahan Rp 312
        * **MAPE (Mean Absolute Percentage Error)**: Kesalahan persentase rata-rata 2.1%
        
        > **Catatan:** Metrik di atas adalah contoh. Nilai aktual bergantung pada data training yang digunakan.
        """)
        
        # Implementation Details
        st.subheader("⚙️ Implementasi Teknis")
        
        tab1, tab2, tab3 = st.tabs(["🐍 Python Libraries", "🔧 Model Training", "🚀 Deployment"])
        
        with tab1:
            st.markdown("""
            **📦 Library yang Digunakan:**
            
            ```python
            # Machine Learning
            import pandas as pd           # Data manipulation
            import numpy as np            # Numerical computing
            import sklearn               # Machine learning toolkit
            
            # Visualization
            import matplotlib.pyplot as plt  # Plotting
            import seaborn as sns           # Statistical visualization
            
            # Web Application
            import streamlit as st          # Web app framework
            import joblib                   # Model serialization
            ```
            
            **🔄 Pipeline Pemrosesan:**
            1. Data loading & validation
            2. Feature engineering
            3. Model training & evaluation
            4. Model serialization
            5. Web deployment
            """)
        
        with tab2:
            st.markdown("""
            **🎯 Proses Training Model:**
            
            ```python
            # 1. Data Preparation
            X = df[['Beras Premium']]  # Feature
            y = df['Beras Medium']     # Target
            
            # 2. Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 3. Model Training
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # 4. Model Evaluation
            y_pred = model.predict(X_test)
            r2_score = model.score(X_test, y_test)
            
            # 5. Model Saving
            joblib.dump(model, 'model_regresi_beras.pkl')
            ```
            """)
        
        with tab3:
            st.markdown("""
            **🚀 Deployment dengan Streamlit:**
            
            ```python
            # 1. Model Loading
            @st.cache_resource
            def load_model():
                return joblib.load("model_regresi_beras.pkl")
            
            # 2. Prediction Function
            def predict_price(premium_price):
                return model.predict([[premium_price]])[0]
            
            # 3. Web Interface
            st.number_input("Harga Premium")
            st.button("Prediksi")
            ```
            
            **📱 Fitur Aplikasi:**
            * Upload CSV untuk batch prediction
            * Manual input untuk single prediction
            * Time series visualization
            * Export hasil prediksi
            """)
        
        # Limitations and Future Work
        st.subheader("⚠️ Keterbatasan & Saran Pengembangan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🚨 Keterbatasan Saat Ini:
            
            * **Model sederhana**: Hanya menggunakan satu fitur
            * **Asumsi linear**: Tidak menangkap pola kompleks
            * **Data terbatas**: Bergantung pada kualitas data training
            * **Faktor eksternal**: Tidak mempertimbangkan inflasi, musim, dll.
            * **Lokasi spesifik**: Model mungkin tidak berlaku universal
            """)
        
        with col2:
            st.markdown("""
            #### 🔮 Pengembangan Mendatang:
            
            * **Multi-feature**: Menambah variabel cuaca, musim, lokasi
            * **Advanced ML**: Random Forest, XGBoost, Neural Networks
            * **Real-time data**: Integrasi dengan API harga pasar
            * **Forecasting**: Prediksi harga untuk periode mendatang
            * **Geographic analysis**: Model per wilayah/provinsi
            """)
        
        # Contact and References
        st.subheader("📚 Referensi & Kontak")
        
        st.markdown("""
        **📖 Referensi Ilmiah:**
        * Gujarati, D. N. (2003). Basic Econometrics. McGraw-Hill.
        * James, G. et al. (2013). An Introduction to Statistical Learning. Springer.
        * McKinney, W. (2017). Python for Data Analysis. O'Reilly Media.
        
        **🛠️ Dokumentasi Teknis:**
        * [Scikit-learn Documentation](https://scikit-learn.org/)
        * [Streamlit Documentation](https://docs.streamlit.io/)
        * [Pandas Documentation](https://pandas.pydata.org/)
        
        ---
        
        💡 **Untuk pertanyaan atau saran pengembangan, silakan hubungi tim pengembang.**
        """)

else:
    st.error("❌ Aplikasi tidak dapat dijalankan karena model tidak berhasil dimuat.")
    st.info("Pastikan file 'model_regresi_beras.pkl' tersedia di direktori yang sama dengan app.py")

# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    © 2025 • Analisis Harga Beras Sumedang • Streamlit Demo<br>
    <em>Dibuat untuk membantu prediksi harga beras di pasar tradisional</em>
</div>
""", unsafe_allow_html=True)