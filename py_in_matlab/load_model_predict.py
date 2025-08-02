# py_in_matlab/load_model_predict.py
import joblib

# Load model (tanpa scaler)
model_knn = joblib.load('model_knn.pkl')
model_svm = joblib.load('model_svm.pkl')

def predict(fitur):
    # Konversi ke array 2D (tanpa normalisasi/scaler)
    fitur_input = [[float(f) for f in fitur]]

    hasil_knn = model_knn.predict(fitur_input)[0]
    hasil_svm = model_svm.predict(fitur_input)[0]

    return [hasil_knn, hasil_svm]
