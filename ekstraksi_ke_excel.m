% Ekstraksi fitur ke file Excel dari dataset train dan test (sehat & sakit)
% Referensi: Feature_Extraction1.m

% Hapus file Excel jika sudah ada
if exist('ekstraksi_fitur_dataset.xlsx', 'file')
    delete('ekstraksi_fitur_dataset.xlsx');
end

% Path dataset
train_sehat_dir = 'dataset/train/sehat/';
train_sakit_dir = 'dataset/train/sakit/';
test_sehat_dir  = 'dataset/test/sehat/';
test_sakit_dir  = 'dataset/test/sakit/';

% List folder dan label
folders = {train_sehat_dir, train_sakit_dir, test_sehat_dir, test_sakit_dir};
labels  = {'sehat', 'sakit', 'sehat', 'sakit'};
groups  = {'train', 'train', 'test', 'test'};

% Inisialisasi tabel hasil untuk train dan test
train_features = [];
test_features = [];
id_counter_train = 1;
id_counter_test = 1;

for i = 1:length(folders)
    folder = folders{i};
    label  = labels{i};
    group  = groups{i};
    files = dir(fullfile(folder, '*.jpg')); % asumsikan gambar .jpg, sesuaikan jika perlu
    
    for j = 1:length(files)
        filename = fullfile(folder, files(j).name);
        img = imread(filename);
        
        % --- Ekstraksi fitur ---
        if size(img,3) == 3
            img_gray = rgb2gray(img);
        else
            img_gray = img;
        end
        
        mean_val = mean(img_gray(:));
        std_val  = std(double(img_gray(:)));
        
        % Ekstraksi fitur 'on' dan 'off' berdasarkan threshold Otsu
        level = graythresh(img_gray); % threshold Otsu (nilai antara 0-1)
        bw = imbinarize(img_gray, level);
        on_val = sum(bw(:)); % jumlah pixel putih (on)
        off_val = numel(bw) - on_val; % jumlah pixel hitam (off)
        
        % Format on_val dan off_val maksimal 4 angka
        on_val = round(on_val, 4, 'significant');
        off_val = round(off_val, 4, 'significant');
        
        % Label: ganti 'sakit' menjadi 'PMK', selain itu 'Normal'
        if strcmp(label, 'sakit')
            label_excel = 'PMK';
        else
            label_excel = 'Normal';
        end
        
        % Simpan ke tabel sesuai urutan header
        if strcmp(group, 'train')
            train_features = [train_features; {id_counter_train, on_val, off_val, mean_val, std_val, label_excel}];
            id_counter_train = id_counter_train + 1;
        else
            test_features = [test_features; {id_counter_test, on_val, off_val, mean_val, std_val, label_excel}];
            id_counter_test = id_counter_test + 1;
        end
    end
end

% Buat tabel dengan header sesuai permintaan
feature_names = {'Id', 'on', 'off', 'mean', 'std', 'label'};
T_train = cell2table(train_features, 'VariableNames', feature_names);
T_test  = cell2table(test_features,  'VariableNames', feature_names);

% Simpan ke file Excel dengan sheet 1 = train, sheet 2 = test
writetable(T_train, 'ekstraksi_fitur_dataset.xlsx', 'Sheet', 'train');
writetable(T_test,  'ekstraksi_fitur_dataset.xlsx', 'Sheet', 'test');

disp('Ekstraksi fitur selesai dan disimpan ke ekstraksi_fitur_dataset.xlsx');
