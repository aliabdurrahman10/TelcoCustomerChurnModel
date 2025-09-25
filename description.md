# Judul Project
Prediksi Customer Churn pada Dataset Telco Menggunakan KNN

## Repository Outline
1. `description.md` - Penjelasan gambaran umum project  
2. `Telco_Churn.ipynb` - Notebook utama yang berisi EDA, preprocessing, training model, evaluasi, dan hyperparameter tuning  
3. `EDA_Telco_Churn.ipynb` - Notebook eksplorasi awal dataset (EDA)  
4. `Inference.ipynb` - Notebook untuk melakukan prediksi (inference) menggunakan model yang sudah disimpan  
5. `model2_best.pkl` - File model terbaik hasil training (KNN)  
6. `WA_Fn-UseC_-Telco-Customer-Churn.csv` - Dataset utama  

## Problem Background
Perusahaan telekomunikasi ingin memprediksi kemungkinan pelanggan berhenti berlangganan (churn). Prediksi ini diharapkan dapat membantu tim marketing untuk mengambil langkah preventif, seperti menawarkan promo atau layanan tambahan kepada pelanggan yang berisiko tinggi churn.  

## User
Tim Manajemen dan Tim Marketing untuk menentukan strategi preventif dalam mencegah pelanggan yang churn.

## Project Output
Model machine learning yang mampu memprediksi apakah pelanggan akan churn atau tidak berdasarkan atribut pelanggan. Model terbaik yang digunakan adalah KNN dengan performa recall yang optimal pada kelas churn.
Lalu hasil dari model kita implementasikan (Deploy) untuk menjalankan inference dari user.

## Data
- **Sumber**: Dataset publik Telco Customer Churn  
- **Jumlah Baris**: 7.043 data pelanggan  
- **Jumlah Kolom**: 21 fitur (fitur kategorikal dan numerik) + target (`Churn`)  
- **Missing Values**: Beberapa kolom memiliki nilai kosong yang telah ditangani  
- **Fitur Utama**: Tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod, dsb.  

## Method
- **Preprocessing**: Handling missing values, encoding data kategorikal, scaling data numerik  
- **Modeling**: Menggunakan beberapa algoritma (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Gradient Boosting)  
- **Evaluasi**: Fokus pada metrik Recall untuk kelas churn (“Yes”)  
- **Model Terbaik**: KNN setelah hyperparameter tuning dan SMOTE 
- **Model Saving & Inference**: Model disimpan dengan `pickle` dan dapat digunakan kembali untuk prediksi di file terpisah  
- **Deployment**: Deploy Program ke Streamlit untuk digunakan user

## Stacks
- **Bahasa Pemrograman**: Python  
- **Library Utama**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Pickle  
- **Tools**: Visual Studio Code
- **Platform**: Streamlit 

## Reference
- [Dataset - Kaggle: Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)  
- Streamlit : https://telcocustomerchurnnn.streamlit.app/
