clc; clear; close all;
%% TIFF FILE
[fnameSEG, pnameSEG] = uigetfile('*.tiff','Please select VESSEL SEGMENTATION file(s)', 'MultiSelect', 'on');
fullnameSEG = strcat(pnameSEG,fnameSEG);
%% Khusus AVRDB
[fnameVES, pnameVES] = uigetfile({'*.jpg'},'Please select VESSEL IMAGE(s)', 'MultiSelect', 'on');
fullnameVES = strcat(pnameVES,fnameVES);
%%
Data = [];
Parameter = [];
for h=14:14
    image = imread(cell2mat(fullnameSEG(h)));
    se = strel('disk', 1);
    Morph = imclose(image,se);
%     Morph = bwareaopen(image, 5);
%     figure, imshowpair(image,Morph);
%     figure, imshow(Morph);

    temp_name = cell2mat(fnameSEG(h));
    temp_name = temp_name(1,5:length(temp_name)-4);
    fout = strcat('Ves_close1', temp_name, '.tiff');
    imwrite(Morph, fout, 'tiff');
    
    vessImage = imread(cell2mat(fullnameVES(h)));
    vessImage = rgb2gray(vessImage);
    T_vI = graythresh(vessImage);
    vI = imbinarize(vessImage, T_vI);
    vI = 1-vI;
   
    idx1 = find(Morph==1);
    TP = length(find(vI(idx1)==1)); FP = length(find(vI(idx1)==0));
    idx2 = find(Morph==0);
    TN = length(find(vI(idx2)==0)); FN = length(find(vI(idx2)==1));
    Parameter = [Parameter; h, TP, FP, TN, FN]
    
    Acc = (TP+TN)/(TP+TN+FP+FN)*100;
    Sen = TP/(TP+FN)*100;
    Spec = TN/(FP+TN)*100;
    Prec = TP/(TP+FP)*100;
    Data = [Data; h, Acc, Sen, Spec, Prec]
end
%% 
    


