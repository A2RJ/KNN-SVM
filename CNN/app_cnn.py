import sys
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import tensorflow as tf

# Import fungsi ekstrak_fitur dari cnn.py
from cnn import ekstrak_fitur, output_model, label_names

# Load model CNN yang sudah dilatih (pastikan model sudah ada, tidak training ulang!)
if not os.path.exists(output_model):
    raise FileNotFoundError(f"❌ File model '{output_model}' tidak ditemukan. Silakan jalankan pelatihan terlebih dahulu.")
model = tf.keras.models.load_model(output_model)

def predict_label(fitur):
    # Normalisasi fitur seperti di cnn.py
    fitur = np.array(fitur, dtype=np.float32)
    # Cek jika semua nilai fitur sama (misal semua nol), hindari pembagian dengan nol
    if np.max(fitur) == 0:
        fitur_norm = fitur
    else:
        fitur_norm = fitur / np.max(fitur)
    fitur_reshaped = fitur_norm.reshape(1, 2, 2, 1)
    pred_proba = model.predict(fitur_reshaped, verbose=0)
    pred_idx = np.argmax(pred_proba, axis=1)[0]
    confidence = pred_proba[0][pred_idx]
    return label_names[pred_idx], confidence, pred_proba[0]

class CNNApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CNN Citra - Uji Gambar")
        self.geometry("500x400")
        self.resizable(False, False)

        self.img_panel = tk.Label(self, text="Belum ada gambar", width=40, height=10, bg="#eee")
        self.img_panel.pack(pady=10)

        self.btn_load = tk.Button(self, text="Pilih Gambar", command=self.load_image)
        self.btn_load.pack(pady=5)

        self.fitur_label = tk.Label(self, text="Fitur: -", font=("Arial", 10))
        self.fitur_label.pack(pady=5)

        self.result_label = tk.Label(self, text="Hasil: -", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=10)

        self.conf_label = tk.Label(self, text="Confidence: -", font=("Arial", 10))
        self.conf_label.pack(pady=5)

        self.detail_label = tk.Label(self, text="", font=("Arial", 9), fg="gray")
        self.detail_label.pack(pady=2)

        self.img_path = None

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return
        self.img_path = file_path
        # Tampilkan gambar
        img = Image.open(file_path)
        img = img.resize((180, 180))
        img_tk = ImageTk.PhotoImage(img)
        self.img_panel.configure(image=img_tk, text="")
        self.img_panel.image = img_tk

        # Ekstrak fitur
        fitur = ekstrak_fitur(file_path)
        self.fitur_label.config(
            text=f"Fitur: On={fitur[0]}, Off={fitur[1]}, Mean={fitur[2]:.3f}, Std={fitur[3]:.3f}"
        )

        # Prediksi label (menggunakan model yang sudah dilatih, tidak training ulang)
        label_pred, confidence, all_probs = predict_label(fitur)
        self.result_label.config(
            text=f"Hasil: {label_pred.upper()}"
        )
        self.conf_label.config(
            text=f"Confidence: {confidence:.3f}"
        )

        # Tampilkan distribusi probabilitas semua kelas untuk debugging
        prob_str = " | ".join([f"{name}:{prob:.2f}" for name, prob in zip(label_names, all_probs)])
        self.detail_label.config(
            text=f"Probabilitas: {prob_str}"
        )

        # Debugging: jika semua hasil selalu "sakit", tampilkan warning
        if all_probs[0] > 0.95 or all_probs[1] > 0.95:
            # Asumsi label_names[0] = "sakit" atau "sehat", tergantung urutan
            # Tampilkan warning jika confidence sangat tinggi pada satu kelas
            print(f"[DEBUG] Probabilitas kelas: {prob_str}")
            if label_pred.lower() == "sakit":
                messagebox.showwarning("Perhatian", "Model selalu memprediksi 'sakit'? Cek data training, distribusi label, dan normalisasi fitur.")

if __name__ == "__main__":
    app = CNNApp()
    app.mainloop()