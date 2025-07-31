clc; clear;

% Parameter
k = 3;
folder_train = 'dataset/train';
folder_test = 'dataset/test';

% --- Konfigurasi untuk ekspor ke Excel ---
output_file = 'ekstraksi_fitur_dataset_4.xlsx';
if exist(output_file, 'file')
    delete(output_file);
end
sheet_names = {'Train', 'Test'};
headers = {'Id', 'On', 'Off', 'Mean', 'Std', 'Label'};

% Fungsi Ekstraksi Fitur
function fitur = ekstrakFitur(img)
    % Grayscale
    gray = rgb2gray(img);

    % Filtering (median filter)
    filtered = medfilt2(gray);

    % Histogram Equalization
    histEq = histeq(filtered);

    % Thresholding
    level = graythresh(histEq);
    bw = imbinarize(histEq, level);

    % Ekstraksi fitur
    fitur_on  = sum(bw(:));
    fitur_off = numel(bw) - fitur_on;
    fitur_mean = mean(bw(:));
    fitur_std  = std(double(bw(:)));

    fitur = [fitur_on, fitur_off, fitur_mean, fitur_std];
end

% === Load Data ===
function [X, y, fitur_rows] = loadDataset(folder)
    kelas = dir(folder);
    kelas = kelas([kelas.isdir] & ~startsWith({kelas.name}, '.'));

    X = [];
    y = [];
    fitur_rows = {};
    idx = 1;
    for i = 1:length(kelas)
        label = kelas(i).name;
        folder_kelas = fullfile(folder, label);
        
        % Ambil semua file dengan ekstensi gambar umum
        ekstensi = {'*.jpg', '*.jpeg', '*.png'};
        gambar = [];
        for e = 1:length(ekstensi)
            gambar = [gambar; dir(fullfile(folder_kelas, ekstensi{e}))];
        end

        for j = 1:length(gambar)
            path = fullfile(folder_kelas, gambar(j).name);
            img = imread(path);
            fitur = ekstrakFitur(img);
            X = [X; fitur];
            y = [y; string(label)];
            fitur_rows(idx, :) = {idx, fitur(1), fitur(2), fitur(3), fitur(4), label}; %#ok<AGROW>
            idx = idx + 1;
        end
    end
end

% Load data train & test, dan siapkan data untuk Excel
[X_train, y_train, fitur_train] = loadDataset(folder_train);
[X_test, y_test, fitur_test] = loadDataset(folder_test);

% ============================
% === Simpan ke Excel ========
% ============================
writecell([headers; fitur_train], output_file, 'Sheet', sheet_names{1}, 'Range', 'A1');
writecell([headers; fitur_test],  output_file, 'Sheet', sheet_names{2}, 'Range', 'A1');
disp("✅ Ekstraksi fitur selesai dan disimpan di 'ekstraksi_fitur_dataset.xlsx'");

% ============================
% === KNN Manual ============
% ============================
y_pred_knn = strings(size(y_test));
for i = 1:size(X_test,1)
    dist = sqrt(sum((X_train - X_test(i,:)).^2, 2));
    [~, idx] = sort(dist);
    k_labels = y_train(idx(1:k));
    y_pred_knn(i) = mode(categorical(k_labels));
end

akurasi_knn = sum(y_pred_knn == y_test) / numel(y_test) * 100;
fprintf("Akurasi KNN manual: %.2f%%\n", akurasi_knn);

% ============================
% === SVM Manual ============
% ============================
y_train_svm = strcmp(y_train, 'sakit') * 2 - 1;
y_test_svm  = strcmp(y_test, 'sakit') * 2 - 1;

% Normalisasi
mean_X = mean(X_train);
std_X  = std(X_train);
X_train_norm = (X_train - mean_X) ./ std_X;
X_test_norm  = (X_test  - mean_X) ./ std_X;

[m, n] = size(X_train_norm);
w = zeros(n,1); b = 0;
alpha = 0.001; lambda = 0.01; epochs = 1000;

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

% Prediksi SVM
pred = X_test_norm * w + b;
y_pred_svm = sign(pred);
y_pred_svm_str = repmat("sehat", size(y_pred_svm));
y_pred_svm_str(y_pred_svm == 1) = "sakit";

akurasi_svm = sum(y_pred_svm == y_test_svm) / numel(y_test_svm) * 100;
fprintf("Akurasi SVM manual: %.2f%%\n", akurasi_svm);

% ============================
% === Simpan model ke MAT ====
% ============================
modelKNN.k = k;
modelKNN.X_train = X_train;
modelKNN.y_train = y_train;

modelSVM.w = w;
modelSVM.b = b;
modelSVM.mean = mean_X;
modelSVM.std  = std_X;

% Hapus file jika sudah ada sebelum menyimpan
if exist('modelKNN.mat', 'file')
    delete('modelKNN.mat');
end
if exist('modelSVM.mat', 'file')
    delete('modelSVM.mat');
end

save('modelKNN.mat', 'modelKNN');
save('modelSVM.mat', 'modelSVM');

fprintf("Model KNN dan SVM berhasil disimpan ke file .mat\n");
