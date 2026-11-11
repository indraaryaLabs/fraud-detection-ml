# Financial Transaction Fraud Detection & Anomaly Identification

## Deskripsi Proyek
Proyek ini adalah implementasi *Machine Learning* untuk mendeteksi penipuan (*fraud detection*) dan mengidentifikasi anomali pada data transaksi keuangan. Mengingat kompleksitas pola transaksi, proyek ini menggunakan pendekatan dua tahap: **Clustering** (Unsupervised Learning) untuk menemukan pola anomali yang tersembunyi, dilanjutkan dengan **Classification** (Supervised Learning) untuk memprediksi anomali pada data transaksi baru.

## Dataset
Dataset yang digunakan berisi **2.512 sampel transaksi** dengan berbagai atribut, antara lain:
- `TransactionAmount`: Nilai transaksi
- `AccountBalance`: Saldo akun nasabah
- `CustomerAge` & `CustomerOccupation`: Profil demografi nasabah
- `Location`, `IP Address`, `DeviceID`, `Channel`: Konteks lokasi dan akses keras transaksi
- `LoginAttempts` & `TransactionDuration`: Perilaku interaksi nasabah saat bertransaksi

## Teknologi & Library yang Digunakan
- **Bahasa Pemrograman:** Python
- **Manipulasi & Analisis Data:** Pandas, NumPy
- **Visualisasi Data:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (K-Means, PCA, Decision Tree, Random Forest, GridSearchCV/RandomizedSearchCV)

## Alur Pemrosesan (Pipeline)
1. **Data Preprocessing:** Pembersihan data, *encoding* fitur kategorikal, dan *scaling* data menggunakan `StandardScaler`.
2. **Clustering (K-Means & PCA):** Menerapkan algoritma K-Means untuk membagi transaksi ke dalam klaster (misalnya: Normal vs Anomali). Evaluasi dilakukan menggunakan *Silhouette Score*.
3. **Classification (Decision Tree & Random Forest):** Hasil klasterisasi digunakan sebagai label target untuk melatih model klasifikasi. Model Random Forest dieksplorasi lebih dalam menggunakan *hyperparameter tuning* agar performanya lebih optimal.

## Struktur Repository
- `[Clustering]_Submission_Akhir_BMLP.ipynb`: Notebook berisi analisis eksplorasi data (EDA) dan pembuatan model K-Means clustering.
- `[Klasifikasi]_Submission_Akhir_BMLP.ipynb`: Notebook berisi pelatihan dan evaluasi model klasifikasi (Decision Tree & Random Forest).
- `data_clustering.csv` & `data_clustering_inverse.csv`: Dataset yang telah diproses dan diberi label klaster.
- `*.h5`: File model *machine learning* yang telah di-training dan siap digunakan (`model_clustering.h5`, `decision_tree_model.h5`, `explore_RandomForest_classification.h5`, dll).