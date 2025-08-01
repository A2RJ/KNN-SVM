# Aplikasi Klasifikasi Gambar dengan KNN dan SVM

Aplikasi ini melakukan klasifikasi gambar menggunakan algoritma K-Nearest Neighbors (KNN) dan Support Vector Machine (SVM) dengan ekstraksi fitur dari citra.

## Struktur Proyek

```
py_in_matlab/
├── app.py                 # Aplikasi utama untuk ekstraksi fitur dan training model
├── test_akurasi.py        # Script untuk testing akurasi model
├── README.md              # File ini
├── model_knn.pkl          # Model KNN (dihasilkan setelah menjalankan app.py)
├── model_svm.pkl          # Model SVM (dihasilkan setelah menjalankan app.py)
├── scaler.pkl             # Scaler untuk normalisasi (dihasilkan setelah menjalankan app.py)
├── ekstraksi_fitur_dataset_4.xlsx  # Dataset hasil ekstraksi fitur
└── app.mlapp              # GUI MATLAB untuk testing model (memuat model dan scaler)

```

## Instalasi Dependencies

Sebelum menjalankan aplikasi, install semua dependencies yang diperlukan:

```bash
pip install opencv-python
pip install numpy
pip install pandas
pip install scikit-learn
pip install joblib
pip install openpyxl
```

Atau install sekaligus dengan requirements.txt:

```bash
pip install -r requirements.txt
```

## Cara Menjalankan

### 1. Menjalankan app.py

Langkah-langkah yang dilakukan oleh `app.py` beserta penjelasan fungsinya:

1. **Ekstraksi Fitur dari Gambar**
   - Script akan membaca seluruh gambar yang ada di folder `../dataset/train` (data latih) dan `../dataset/test` (data uji).
   - Untuk setiap gambar, dilakukan beberapa tahapan preprocessing:
     - **Konversi ke grayscale:** Mengubah gambar berwarna menjadi hitam-putih agar lebih sederhana untuk diproses.
     - **Median blur:** Mengurangi noise pada gambar dengan filter median.
     - **Histogram equalization:** Meningkatkan kontras gambar agar fitur lebih mudah dikenali.
     - **Thresholding Otsu:** Mengubah gambar grayscale menjadi gambar biner (hitam-putih) secara otomatis berdasarkan nilai threshold optimal.
   - Setelah preprocessing, dari setiap gambar diekstrak 4 fitur utama:
     - **On:** Jumlah piksel aktif (bernilai 1) pada gambar biner.
     - **Off:** Jumlah piksel tidak aktif (bernilai 0) pada gambar biner.
     - **Mean:** Rata-rata nilai piksel pada gambar biner.
     - **Std:** Standar deviasi nilai piksel pada gambar biner.

2. **Menyimpan Hasil Ekstraksi ke File Excel**
   - Semua fitur yang diekstrak dari gambar beserta labelnya disimpan ke file Excel `ekstraksi_fitur_dataset_4.xlsx`.
   - Data train dan test disimpan pada sheet yang berbeda agar mudah dianalisis dan digunakan kembali.

3. **Melatih Model KNN dan SVM**
   - Fitur dari data train dinormalisasi menggunakan StandardScaler agar setiap fitur memiliki skala yang sama.
   - Model K-Nearest Neighbors (KNN) dengan k=3 dilatih menggunakan data train yang sudah dinormalisasi.
   - Model Support Vector Machine (SVM) dengan kernel linear juga dilatih pada data yang sama.

4. **Menyimpan Model yang Sudah Dilatih**
   - Setelah proses training selesai, model KNN, model SVM, dan scaler normalisasi disimpan ke file (`model_knn.pkl`, `model_svm.pkl`, dan `scaler.pkl`).
   - File-file ini nantinya dapat digunakan untuk melakukan prediksi pada data baru tanpa perlu melakukan training ulang.

Dengan demikian, menjalankan `app.py` akan menyiapkan seluruh pipeline mulai dari ekstraksi fitur, penyimpanan dataset, training model, hingga penyimpanan model siap pakai.

```bash
python app.py
```

### 2. Menjalankan test_akurasi.py

Script ini akan:
- Memuat model yang sudah dilatih
- Melakukan prediksi pada data test
- Menampilkan hasil akurasi, confusion matrix, dan classification report

```bash
python test_akurasi.py
```

### 3. Menggunakan GUI MATLAB (app.mlapp)

File `../app.mlapp` adalah aplikasi GUI MATLAB yang dapat digunakan untuk testing model secara interaktif:

- **Memuat Model:** GUI akan memuat model KNN, SVM, dan scaler yang sudah dilatih dari file `.pkl`
- **Input Gambar:** Pengguna dapat memilih gambar untuk diklasifikasi melalui interface GUI
- **Processing:** Sistem akan melakukan ekstraksi fitur dan prediksi secara otomatis
- **Hasil Klasifikasi:** Menampilkan hasil klasifikasi (sehat/sakit)

**Cara menggunakan:**
1. Pastikan model sudah dilatih dengan menjalankan `app.py` terlebih dahulu
2. Buka file `app.mlapp` di MATLAB
3. Jalankan aplikasi GUI
4. Pilih gambar yang ingin diklasifikasi
5. Lihat hasil prediksi yang ditampilkan

## Output yang Dihasilkan

Setelah menjalankan `app.py`, akan dihasilkan file-file berikut:

1. **ekstraksi_fitur_dataset_4.xlsx** - Dataset dengan fitur yang diekstrak
2. **model_knn.pkl** - Model KNN yang sudah dilatih
3. **model_svm.pkl** - Model SVM yang sudah dilatih
4. **scaler.pkl** - Scaler untuk normalisasi data

## Fitur yang Diekstrak

Aplikasi mengekstrak 4 fitur dari setiap gambar:
- **On** - Jumlah piksel yang aktif (bernilai 1)
- **Off** - Jumlah piksel yang tidak aktif (bernilai 0)
- **Mean** - Rata-rata nilai piksel
- **Std** - Standar deviasi nilai piksel

## Preprocessing Gambar

Sebelum ekstraksi fitur, gambar akan diproses dengan:
1. Konversi ke grayscale
2. Median blur untuk mengurangi noise
3. Histogram equalization untuk meningkatkan kontras
4. Thresholding Otsu untuk binarisasi

## Klasifikasi

Aplikasi menggunakan 2 algoritma klasifikasi:
- **KNN** dengan k=3
- **SVM** dengan kernel linear

## Troubleshooting

### Error: "No module named 'openpyxl'"
```bash
pip install openpyxl
```

### Error: "No module named 'cv2'"
```bash
pip install opencv-python
```

### Error: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

## Catatan

- Pastikan folder `../dataset/train` dan `../dataset/test` berisi gambar yang akan diklasifikasi
- Format gambar yang didukung: .jpg, .jpeg, .png
- Struktur folder dataset harus memiliki subfolder untuk setiap kelas (misal: sehat, sakit) 