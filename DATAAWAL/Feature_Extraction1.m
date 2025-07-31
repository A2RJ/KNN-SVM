clc, clear all;
file_excl ='/Users/siskaatmawanoktavia/Downloads/THESIS EDITING /APP DIABETIC RETINOPATHY/data training.xlsx';
%'/Users/siskaatmawanoktavia/Documents/APP DIABETIC RETINOPATHY/data siska.xlsx';     
label = 'DR'; % pilih antara Normal atau Sakit
sheet='Train';
folder = '/Users/siskaatmawanoktavia/Downloads/THESIS EDITING /APP DIABETIC RETINOPATHY/Images/UJI/TRAIN 80%';
%'/Users/siskaatmawanoktavia/Documents/APP DIABETIC RETINOPATHY/Images/UJI/TRAIN 80%/';
folder = strcat(folder,label,'/*.jpg');
img_files = dir(folder);

kolom=64; baris=64;

nfiles = length(img_files);    % Number of files found
fprintf ("jumlah data : %i",nfiles)

start = 54; % ganti nilainya sesuai baris terahir data yang ada di excel
for i=1:(nfiles)
    path = strcat(img_files(i).folder,'/',img_files(i).name);
    img = imread(path);
    img = rgb2gray (img);
    img = imresize(img,[baris kolom]);
    M = mean2(img);
    std = std2(img);
    [on_pxl,off_pxl]=count_img(img);
      total = bwarea(img);
    row=start-1+i;
    id=row-1;
    writecell({id;},file_excl, 'Sheet',sheet,'Range', strcat("A",num2str(row))) %nyimpen data standar deviasi ke excel
    writecell({on_pxl;},file_excl, 'Sheet',sheet,'Range', strcat("B",num2str(row))) %nyimpen data id ke excel
    writecell({off_pxl;},file_excl, 'Sheet',sheet,'Range', strcat("C",num2str(row))) %nyimpen data jumlah pixel on ke excel
    writecell({M;},file_excl, 'Sheet',sheet,'Range', strcat("D",num2str(row))) %nyimpen data jumlah pixel off ke excel
    writecell({std;},file_excl, 'Sheet',sheet,'Range', strcat("E",num2str(row))) %nyimpen data rata2 on ke excel
    writecell({label;},file_excl, 'Sheet',sheet,'Range', strcat("F",num2str(row))) %nyimpen label ke excel
end

function [on_pxl,off_pxl] = count_img(img)
    th= 55; on_pxl=0; off_pxl=0;
    [baris,kolom] = size(img); 
    for i=1:baris
        for j=1:kolom
            if img(i,j)>=th
                on_pxl=on_pxl + 1;
            else
                off_pxl=off_pxl + 1;
            
            end
        end
    end   
end

