clc; clear;

% Lokasi penyimpanan hasil Excel
output_file = 'ekstraksi_fitur_dataset.xlsx';
if exist(output_file, 'file')
    delete(output_file);
end

% Konfigurasi
sheet_names = {'Train', 'Test'};
groups = {'train', 'test'};
labels = {'sehat', 'sakit'};
baris = 64; kolom = 64;
threshold = 55;

% Header kolom
headers = {'Id', 'On', 'Off', 'Mean', 'Std', 'Label'};

% Loop untuk masing-masing sheet (Train dan Test)
for s = 1:2
    sheet = sheet_names{s};
    data_rows = {};  % buffer penyimpanan baris data
    idx = 1;
    
    for l = 1:2  % sehat dan sakit
        label = labels{l};
        folder = fullfile('dataset', groups{s}, label);
        img_files = dir(fullfile(folder, '*.jpg'));
        img_files = [img_files; dir(fullfile(folder, '*.jpeg'))];
        
        for i = 1:length(img_files)
            path = fullfile(img_files(i).folder, img_files(i).name);
            img = imread(path);
            if size(img,3) == 3
                img = rgb2gray(img);
            end
            img = imresize(img, [baris kolom]);

            % Ekstraksi fitur
            mean_val = mean2(img);
            std_val = std2(double(img));
            [on_pxl, off_pxl] = count_img(img, threshold);

            % Simpan ke buffer baris
            data_rows(idx, :) = {idx, on_pxl, off_pxl, mean_val, std_val, label}; %#ok<AGROW>
            idx = idx + 1;
        end
    end

    % Tulis ke Excel (Header + Data)
    writecell([headers; data_rows], output_file, 'Sheet', sheet, 'Range', 'A1');
end

disp("✅ Ekstraksi selesai dan disimpan di 'ekstraksi_fitur_dataset.xlsx'");

% --- Fungsi bantu hitung ON/OFF ---
function [on_pxl, off_pxl] = count_img(img, th)
    bw = img >= th;
    on_pxl = sum(bw(:));
    off_pxl = numel(bw) - on_pxl;
end