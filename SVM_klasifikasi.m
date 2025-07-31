clc; clear;

% === Load Data ===
file_excel = 'ekstraksi_fitur_dataset.xlsx';
train_data = readtable(file_excel, 'Sheet', 'Train');
test_data  = readtable(file_excel, 'Sheet', 'Test');

% Ambil fitur dan label (4 fitur)
X_train = table2array(train_data(:,2:5)); % ON, OFF, Mean, Std
y_train = strcmp(train_data.Label, 'sakit')*2 - 1;  % sakit=+1, sehat=-1

X_test = table2array(test_data(:,2:5));
y_test_str = test_data.Label;
y_test = strcmp(y_test_str, 'sakit')*2 - 1;

% === Normalisasi (opsional tapi disarankan) ===
X_train = normalize(X_train);
X_test = normalize(X_test);

% === Inisialisasi Parameter ===
[m, n] = size(X_train);
w = zeros(n,1);      % bobot
b = 0;               % bias
alpha = 0.001;       % learning rate
lambda = 0.01;       % regularisasi
epochs = 1000;

% === Training dengan Gradient Descent ===
for epoch = 1:epochs
    for i = 1:m
        if y_train(i) * (X_train(i,:) * w + b) < 1
            w = w - alpha * (2*lambda*w - y_train(i)*X_train(i,:)');
            b = b + alpha * y_train(i);
        else
            w = w - alpha * 2 * lambda * w;
        end
    end
end

% === Prediksi ===
pred = X_test * w + b;
y_pred = sign(pred);

% === Konversi hasil ke label
y_pred_str = repmat("sehat", size(y_pred));
y_pred_str(y_pred == 1) = "sakit";

% === Evaluasi ===
akurasi = sum(y_pred == y_test) / length(y_test) * 100;
fprintf("Akurasi SVM Manual (tanpa toolbox): %.2f%%\n", akurasi);

% === Tampilkan hasil
hasil = table(test_data.Id, y_test_str, y_pred_str, 'VariableNames', {'Id', 'LabelAsli', 'Prediksi'});
disp(hasil);
