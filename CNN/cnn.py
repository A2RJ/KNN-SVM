import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# --- Konfigurasi ---
train_folder = '../dataset/train'
test_folder = '../dataset/test'
output_excel = 'cnn_ekstraksi_fitur.xlsx'
sheet_train = 'Train'
sheet_test = 'Test'
output_model = 'cnn_trained_model.h5'  # Nama file untuk menyimpan model

# --- Ekstraksi fitur dari citra (sama dengan app.py) ---
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

# --- Load dataset dari folder (sama dengan app.py) ---
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

# --- Hapus file Excel jika sudah ada ---
try:
    if os.path.exists(output_excel):
        os.remove(output_excel)
        print(f"🗑️ File {output_excel} lama dihapus")
except PermissionError:
    print(f"⚠️ File {output_excel} sedang digunakan, akan ditimpa")

# --- Proses dan Simpan (sama dengan app.py) ---
X_train, y_train, fitur_train = load_dataset(train_folder)
X_test, y_test, fitur_test = load_dataset(test_folder)

# Simpan ke Excel dengan format yang sama
headers = ['Id', 'On', 'Off', 'Mean', 'Std', 'Label']
df_train = pd.DataFrame(fitur_train, columns=headers)
df_test = pd.DataFrame(fitur_test, columns=headers)
with pd.ExcelWriter(output_excel) as writer:
    df_train.to_excel(writer, sheet_name=sheet_train, index=False)
    df_test.to_excel(writer, sheet_name=sheet_test, index=False)
print(f"✅ Data disimpan ke {output_excel}")

# --- Normalisasi untuk CNN ---
X_train_norm = X_train.astype(np.float32) / np.max(X_train)
X_test_norm = X_test.astype(np.float32) / np.max(X_test)

# --- Reshape untuk CNN (batch_size, height, width, channels) ---
X_train_cnn = X_train_norm.reshape(-1, 2, 2, 1)  # Reshape 4 fitur menjadi 2x2
X_test_cnn = X_test_norm.reshape(-1, 2, 2, 1)

# --- Split train/val ---
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_cnn, y_train, test_size=0.15, random_state=42, stratify=y_train
)

# --- Buat label numerik ---
label_names = sorted(list(set(y_train)))
label_to_idx = {name: idx for idx, name in enumerate(label_names)}
y_train_num = np.array([label_to_idx[label] for label in y_train_split])
y_val_num = np.array([label_to_idx[label] for label in y_val_split])
y_test_num = np.array([label_to_idx[label] for label in y_test])

# --- CNN Model (disesuaikan untuk input 2x2) ---
model = models.Sequential([
    layers.Conv2D(16, (2, 2), activation='relu', input_shape=(2, 2, 1)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(len(label_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("🔄 Training CNN model...")
# --- Training CNN ---
history = model.fit(
    X_train_split, y_train_num,
    epochs=50,
    batch_size=8,
    validation_data=(X_val_split, y_val_num),
    verbose=1
)

# --- Simpan model hasil training ---
model.save(output_model)
print(f"💾 Model hasil training disimpan ke '{output_model}'")

# --- Evaluasi pada data train ---
train_loss, train_acc = model.evaluate(X_train_split, y_train_num, verbose=0)
val_loss, val_acc = model.evaluate(X_val_split, y_val_num, verbose=0)

# --- Prediksi pada data test ---
y_test_pred_proba = model.predict(X_test_cnn)
y_test_pred = np.argmax(y_test_pred_proba, axis=1)

# --- Simpan hasil prediksi ke Excel baru ---
results_test = []
for idx, (true_label, pred_label) in enumerate(zip(y_test, y_test_pred)):
    true_label_name = label_names[true_label] if isinstance(true_label, int) else true_label
    pred_label_name = label_names[pred_label]
    confidence = y_test_pred_proba[idx][pred_label]
    results_test.append([idx+1, true_label_name, pred_label_name, f"{confidence:.3f}"])

df_results_test = pd.DataFrame(results_test, columns=['Id', 'LabelAsli', 'Prediksi', 'Confidence'])

# --- Simpan hasil prediksi ke sheet baru ---
with pd.ExcelWriter(output_excel, mode='a', if_sheet_exists='replace') as writer:
    df_results_test.to_excel(writer, sheet_name='CNN_Prediksi', index=False)

print(f"✅ Hasil prediksi CNN disimpan ke sheet 'CNN_Prediksi'")
print(f"📊 Akurasi train: {train_acc*100:.2f}%")
print(f"📊 Akurasi validation: {val_acc*100:.2f}%")

# --- Hitung akurasi test ---
correct = sum(1 for true, pred in zip(y_test, y_test_pred) 
             if (label_names[true] if isinstance(true, int) else true) == label_names[pred])
test_acc = correct / len(y_test)
print(f"📊 Akurasi test: {test_acc*100:.2f}%")