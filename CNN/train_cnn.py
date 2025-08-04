# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # === Image Parameters ===
# img_size = (100, 100)
# batch_size = 16

# # === Augmentasi dan Normalisasi ===
# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen  = ImageDataGenerator(rescale=1./255)

# train_gen = train_datagen.flow_from_directory(
#     'dataset/train',
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='binary'
# )

# test_gen = test_datagen.flow_from_directory(
#     'dataset/test',
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=False
# )

# # === CNN Model ===
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
#     MaxPooling2D(2,2),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # === Train ===
# model.fit(train_gen, epochs=10, validation_data=test_gen)

# # === Simpan model ===
# model.save('cnn_model.h5')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from scipy.io import savemat
import numpy as np

# === Image Parameters ===
img_size = (100, 100)
batch_size = 16

# === Augmentasi dan Normalisasi ===
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_gen = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# === CNN Model ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === Train ===
model.fit(train_gen, epochs=10, validation_data=test_gen)

# === Simpan model ===
model.save('cnn_model.h5')

# === Evaluasi pada data test ===
y_true = test_gen.classes  # label asli (0 = sehat, 1 = sakit)
y_pred_prob = model.predict(test_gen).flatten()
y_pred_label = (y_pred_prob >= 0.5).astype(int)  # thresholding

akurasi = accuracy_score(y_true, y_pred_label)

# === Simpan ke .mat file ===
savemat('cnn_results.mat', {
    'y_true': y_true,
    'y_pred': y_pred_label,
    'y_prob': y_pred_prob,
    'accuracy': akurasi
})

print(f"✅ Model disimpan sebagai cnn_model.h5 dan hasil prediksi disimpan sebagai cnn_results.mat dengan akurasi {akurasi * 100:.2f}%")
