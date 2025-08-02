import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Konfigurasi ---
output_excel = 'cnn_ekstraksi_fitur.xlsx'
sheet_test = 'Test'

# --- Load data test dari sheet Test ---
df_test = pd.read_excel(output_excel, sheet_name=sheet_test)

# Ambil fitur dan label
X_test = df_test[['On', 'Off', 'Mean', 'Std']].values.astype(np.float32)
y_test = df_test['Label'].values

# --- Normalisasi seperti di cnn.py ---
X_test_norm = X_test / np.max(X_test)

# --- Reshape untuk CNN (batch_size, height, width, channels) ---
X_test_cnn = X_test_norm.reshape(-1, 2, 2, 1)

# --- Load label mapping dari data train (sheet Train) ---
df_train = pd.read_excel(output_excel, sheet_name='Train')
label_names = sorted(list(set(df_train['Label'])))
label_to_idx = {name: idx for idx, name in enumerate(label_names)}
y_test_num = np.array([label_to_idx[label] for label in y_test])

# --- Definisikan ulang model CNN sesuai cnn.py ---
model = models.Sequential([
    layers.Conv2D(16, (2, 2), activation='relu', input_shape=(2, 2, 1)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(len(label_names), activation='softmax')
])

# --- Load bobot model dari file jika ada ---
if os.path.exists('cnn_trained_model.h5'):
    model.load_weights('cnn_trained_model.h5')
else:
    print("❌ File bobot cnn_trained_model.h5 tidak ditemukan. Silakan jalankan cnn.py untuk melatih dan menyimpan model.")
    exit(1)

# --- Prediksi pada data test ---
y_test_pred_proba = model.predict(X_test_cnn)
y_test_pred = np.argmax(y_test_pred_proba, axis=1)

# --- Hitung akurasi test ---
correct = sum(1 for true, pred in zip(y_test, y_test_pred)
              if (label_names[true] if isinstance(true, int) else true) == label_names[pred])
test_acc = correct / len(y_test)
print(f"📊 Akurasi test: {test_acc*100:.2f}%")
