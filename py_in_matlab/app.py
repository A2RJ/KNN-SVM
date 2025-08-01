# py_in_matlab/app.py
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# --- Konfigurasi ---
train_folder = '../dataset/train'
test_folder = '../dataset/test'
output_excel = 'ekstraksi_fitur_dataset_4.xlsx'
sheet_train = 'Train'
sheet_test = 'Test'

# --- Ekstraksi fitur dari citra ---
def ekstrak_fitur(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.medianBlur(gray, 3)
    hist_eq = cv2.equalizeHist(filtered)
    _, bw = cv2.threshold(hist_eq, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fitur_on = np.sum(bw)
    fitur_off = bw.size - fitur_on
    fitur_mean = np.mean(bw)
    fitur_std = np.std(bw.astype(np.float32))

    return [fitur_on, fitur_off, fitur_mean, fitur_std]

# --- Load dataset dari folder ---
def load_dataset(folder):
    data = []
    labels = []
    rows = []
    idx = 1
    for label in os.listdir(folder):
        path_label = os.path.join(folder, label)
        if os.path.isdir(path_label):
            for file in os.listdir(path_label):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(path_label, file)
                    fitur = ekstrak_fitur(path)
                    data.append(fitur)
                    labels.append(label)
                    rows.append([idx] + fitur + [label])
                    idx += 1
    return np.array(data), np.array(labels), rows

# --- Proses dan Simpan ---
X_train, y_train, fitur_train = load_dataset(train_folder)
X_test, y_test, fitur_test = load_dataset(test_folder)

# Simpan ke Excel
headers = ['Id', 'On', 'Off', 'Mean', 'Std', 'Label']
df_train = pd.DataFrame(fitur_train, columns=headers)
df_test = pd.DataFrame(fitur_test, columns=headers)
with pd.ExcelWriter(output_excel) as writer:
    df_train.to_excel(writer, sheet_name=sheet_train, index=False)
    df_test.to_excel(writer, sheet_name=sheet_test, index=False)
print(f"✅ Data disimpan ke {output_excel}")

# --- Normalisasi ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train model ---
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)

# --- Simpan model dan scaler ---
joblib.dump(knn, 'model_knn.pkl')
joblib.dump(svm, 'model_svm.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model KNN, SVM, dan scaler berhasil disimpan.")
