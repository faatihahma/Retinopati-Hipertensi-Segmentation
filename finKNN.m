clc; clear;
tic
load('E:\Image AVRDB kelas\image\OD kanan\ektraksi fitur\Data_5_exud.mat');
X =  tabel(:, 3:17);
Y =  tabel(:, 18);

Mdl = fitcknn(X,Y,'NumNeighbors',30);  %fitcsvm(X,Y) default 30;
fout = strcat('KNN_5_exud_30.mat');
save(fout,'Mdl');
toc
