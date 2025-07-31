clear
clc

folder = "/Users/siskaatmawanoktavia/Downloads/PMK.jpeg"
th= 55
kolom=8
baris=8
on_pxl=0
off_pxl=0

img = imread (folder);
img=rgb2gray(img)
img=imresize(img,[baris kolom])
M=mean(img,'all')
std=std2(img)

for i=1:baris
    for k=1:kolom
        if img(i,k)>= th
            on_pxl = on_pxl+1
        else
            off_pxl = off_pxl+1`
        end
    end     
end
clc
disp(img)
fprintf("nilai array : %i\n", img(1,2))
fprintf("jumlah pixel on : %i\n", on_pxl)
fprintf("jumlah pixel off : %i\n", off_pxl)
fprintf("rata-rata : %i\n", M)
fprintf("standar deviasi : %i\n", std)