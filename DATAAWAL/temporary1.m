clc, clear all;


load_model = '/Users/siskaatmawanoktavia/Documents/APP DIABETIC RETINOPATHY/KNN_Classification_data_siska.mat';
path = '/Users/siskaatmawanoktavia/Documents/APP DIABETIC RETINOPATHY/Normal1.jpeg';
kolom=64; baris=64;

% x_test(i,[1 2 3 4]

img = imread(path);
img = rgb2gray (img);
img = imresize(img,[baris kolom]);
Mean = mean2(img);
Std = std2(img);
[On,Off]=count_img(img);

dt_img = table (On,Off,Mean,Std);
disp(class (dt_img))

model = loadLearnerForCoder(load_model);
label = predict(model,dt_img);
disp (label)
    
function [On,Off] = count_img(img)
    th= 55; On=0; Off=0;
    [baris,kolom] = size(img); 
    for i=1:baris
        for j=1:kolom
            if img(i,j)>=th
                On=On + 1;
            else
                Off=Off + 1;
              
            end
        end
    end   
end

