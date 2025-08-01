import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# === Load data dari Excel ===
df_train = pd.read_excel('../image_processing/ekstraksi_fitur_dataset_4.xlsx', sheet_name='Train')
df_test = pd.read_excel('../image_processing/ekstraksi_fitur_dataset_4.xlsx', sheet_name='Test')

# Ambil fitur dan label
X_train = df_train[['On', 'Off', 'Mean', 'Std']].values
y_train = df_train['Label'].values

X_test = df_test[['On', 'Off', 'Mean', 'Std']].values
y_test = df_test['Label'].values

# === Train dan simpan model ===
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Simpan model
joblib.dump(knn, 'model_knn.pkl')
joblib.dump(svm, 'model_svm.pkl')

print("✅ Model KNN dan SVM berhasil dibuat dan disimpan.")
