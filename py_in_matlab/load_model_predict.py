# py_in_matlab/load_model_predict.py
import joblib

# Load model dan scaler (hanya sekali)
model_knn = joblib.load('model_knn.pkl')
model_svm = joblib.load('model_svm.pkl')
scaler = joblib.load('scaler.pkl')

def predict(fitur):
    # Konversi ke array 2D dan normalisasi
    fitur_input = [[float(f) for f in fitur]]
    fitur_scaled = scaler.transform(fitur_input)

    hasil_knn = model_knn.predict(fitur_scaled)[0]
    hasil_svm = model_svm.predict(fitur_scaled)[0]

    return [hasil_knn, hasil_svm]
