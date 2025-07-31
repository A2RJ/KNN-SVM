% KNN_klasifikasi.m
clc; clear;

% Baca data dari file Excel
filename = 'ekstraksi_fitur_dataset.xlsx';  % Ganti jika nama file berbeda
train_data = readtable(filename, 'Sheet', 'Train');
test_data  = readtable(filename, 'Sheet', 'Test');

% Ekstrak fitur dan label dari data latih
X_train = table2array(train_data(:, {'On', 'Off', 'Mean', 'Std'}));
y_train = train_data.Label;

% Ekstrak fitur dan label dari data uji
X_test = table2array(test_data(:, {'On', 'Off', 'Mean', 'Std'}));
y_test = test_data.Label;

% Buat model KNN (default K = 5)
modelKNN = fitcknn(X_train, y_train, 'NumNeighbors', 5);

% Prediksi data uji
y_pred = predict(modelKNN, X_test);

% Evaluasi hasil
confmat = confusionmat(y_test, y_pred);
akurasi = sum(strcmp(y_test, y_pred)) / length(y_test) * 100;

% Tampilkan hasil
disp('Confusion Matrix:');
disp(confmat);

fprintf('Akurasi: %.2f%%\n', akurasi);
