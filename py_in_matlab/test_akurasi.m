% test_akurasi.m
% Menghitung akurasi model KNN dan SVM dari file Excel menggunakan Python

% Pastikan environment Python sudah terhubung
if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),''); 
end
py.importlib.import_module('load_model_predict');  % Pastikan module py ada

% Load data dari Excel (sheet Test)
file_excel = '../image_processing/ekstraksi_fitur_dataset_4.xlsx';
opts = detectImportOptions(file_excel, 'Sheet', 'Test');
opts = setvartype(opts, {'On','Off','Mean','Std'}, 'double');
data = readtable(file_excel, opts, 'Sheet', 'Test');

% Ekstrak fitur dan label
X_test = table2array(data(:, {'On','Off','Mean','Std'}));
y_true = string(data.Label);

% Inisialisasi array prediksi
y_pred_knn = strings(height(data), 1);
y_pred_svm = strings(height(data), 1);

% Lakukan prediksi menggunakan model Python
for i = 1:height(data)
    fitur = X_test(i,:);
    [knn_label, svm_label] = load_model_predict(fitur);
    y_pred_knn(i) = string(knn_label);
    y_pred_svm(i) = string(svm_label);
end

% Hitung akurasi
akurasi_knn = sum(y_pred_knn == y_true) / numel(y_true) * 100;
akurasi_svm = sum(y_pred_svm == y_true) / numel(y_true) * 100;

fprintf("✅ Akurasi KNN (Python): %.2f%%\n", akurasi_knn);
fprintf("✅ Akurasi SVM (Python): %.2f%%\n", akurasi_svm);
