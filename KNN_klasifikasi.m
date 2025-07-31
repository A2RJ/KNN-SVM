clc; clear;

% === Parameter ===
k = 3;
file_excel = 'ekstraksi_fitur_dataset.xlsx';  % Ganti dengan nama file Excel kamu

% === Baca data ===
train_data = readtable(file_excel, 'Sheet', 'Train');
test_data  = readtable(file_excel, 'Sheet', 'Test');

X_train = table2array(train_data(:,2:5));  % ON, OFF, Mean, Std
y_train = train_data.Label;

X_test = table2array(test_data(:,2:5));
y_test = test_data.Label;

% === KNN manual ===
y_pred = strings(size(y_test));

for i = 1:size(X_test,1)
    dist = sqrt(sum((X_train - X_test(i,:)).^2, 2));
    [~, idx] = sort(dist);
    k_labels = y_train(idx(1:k));
    y_pred(i) = mode(categorical(k_labels));
end

% === Evaluasi ===
akurasi = sum(y_pred == y_test) / numel(y_test) * 100;
fprintf('Akurasi KNN manual: %.2f%%\n', akurasi);

hasil = table(test_data.Id, y_test, y_pred, 'VariableNames', {'Id', 'LabelAsli', 'Prediksi'});
disp(hasil);
