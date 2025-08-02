# py_in_matlab/test_akurasi.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === Load dataset dari file Excel ===
df_test = pd.read_excel('ekstraksi_fitur_dataset_4.xlsx', sheet_name='Test')
X_test = df_test[['On', 'Off', 'Mean', 'Std']].values
y_test = df_test['Label'].values

# === Load model (tanpa scaler) ===
knn = joblib.load('model_knn.pkl')
svm = joblib.load('model_svm.pkl')

# === Prediksi dan evaluasi (tanpa normalisasi/scaler) ===
y_pred_knn = knn.predict(X_test)
y_pred_svm = svm.predict(X_test)

# === Hasil evaluasi ===
print("=== Evaluasi KNN ===")
print("Akurasi :", accuracy_score(y_test, y_pred_knn) * 100, "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

print("\n=== Evaluasi SVM ===")
print("Akurasi :", accuracy_score(y_test, y_pred_svm) * 100, "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
