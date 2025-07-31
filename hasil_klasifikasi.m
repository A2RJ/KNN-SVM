clc; clear;

% === Parameter ===
k = 3;
file_excel_fitur = 'ekstraksi_fitur_dataset.xlsx';  % File fitur
file_excel_hasil = 'hasil_klasifikasi.xlsx';        % File hasil klasifikasi

% Jika file hasil sudah ada, hapus dulu
if exist(file_excel_hasil, 'file')
    delete(file_excel_hasil);
end

% === Baca data fitur ===
train_data = readtable(file_excel_fitur, 'Sheet', 'Train');
test_data  = readtable(file_excel_fitur, 'Sheet', 'Test');

X_train = table2array(train_data(:,2:5));  % ON, OFF, Mean, Std
y_train = train_data.Label;

X_test = table2array(test_data(:,2:5));
y_test = test_data.Label;

%% === KNN Manual ===
y_pred_knn = strings(size(y_test));
for i = 1:size(X_test,1)
    dist = sqrt(sum((X_train - X_test(i,:)).^2, 2));
    [~, idx] = sort(dist);
    k_labels = y_train(idx(1:k));
    y_pred_knn(i) = mode(categorical(k_labels));
end

% Evaluasi KNN
akurasi_knn = sum(y_pred_knn == y_test) / numel(y_test) * 100;
fprintf('Akurasi KNN manual: %.2f%%\n', akurasi_knn);

% Simpan ke Excel - Sheet KNN
hasil_knn = table(test_data.Id, y_test, y_pred_knn, ...
    'VariableNames', {'Id', 'LabelAsli', 'Prediksi'});
writetable(hasil_knn, file_excel_hasil, 'Sheet', 'KNN');

%% === SVM Manual ===
% Ubah label ke numerik (-1, +1)
y_train_svm = strcmp(y_train, 'sakit')*2 - 1;
y_test_svm  = strcmp(y_test, 'sakit')*2 - 1;

% Normalisasi
X_train_norm = normalize(X_train);
X_test_norm  = normalize(X_test);

% Parameter SVM
[m, n] = size(X_train_norm);
w = zeros(n,1); b = 0;
alpha = 0.001; lambda = 0.01; epochs = 1000;

% Training SVM manual (Gradient Descent)
for epoch = 1:epochs
    for i = 1:m
        if y_train_svm(i) * (X_train_norm(i,:) * w + b) < 1
            w = w - alpha * (2*lambda*w - y_train_svm(i)*X_train_norm(i,:)');
            b = b + alpha * y_train_svm(i);
        else
            w = w - alpha * 2 * lambda * w;
        end
    end
end

% Prediksi
pred = X_test_norm * w + b;
y_pred_svm = sign(pred);

% Konversi label
y_pred_svm_str = repmat("sehat", size(y_pred_svm));
y_pred_svm_str(y_pred_svm == 1) = "sakit";

% Evaluasi
akurasi_svm = sum(y_pred_svm == y_test_svm) / numel(y_test_svm) * 100;
fprintf("Akurasi SVM Manual: %.2f%%\n", akurasi_svm);

% Simpan ke Excel - Sheet SVM
hasil_svm = table(test_data.Id, y_test, y_pred_svm_str, ...
    'VariableNames', {'Id', 'LabelAsli', 'Prediksi'});
writetable(hasil_svm, file_excel_hasil, 'Sheet', 'SVM');

%% === Simpan Akurasi ke Sheet ke-3 ===
nama_metode = ["KNN"; "SVM"];
akurasi = [akurasi_knn; akurasi_svm];
tabel_akurasi = table(nama_metode, akurasi, 'VariableNames', {'Metode', 'Akurasi'});
writetable(tabel_akurasi, file_excel_hasil, 'Sheet', 3);
