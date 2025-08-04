import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load model
model = load_model('cnn_model.h5')

def predict_image(image_path):
    img = load_img(image_path, target_size=(100, 100))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    label = "sehat" if pred >= 0.5 else "sakit"
    return label, pred

def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if file_path:
        label, prob = predict_image(file_path)
        result_var.set(f"Prediksi: {label}, Probabilitas: {prob:.4f}")

# GUI setup
root = tk.Tk()
root.title("Prediksi Kesehatan Daun (CNN)")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

btn = tk.Button(frame, text="Pilih Gambar", command=open_file)
btn.pack(pady=10)

result_var = tk.StringVar()
result_label = tk.Label(frame, textvariable=result_var, font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
