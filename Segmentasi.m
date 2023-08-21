%% PEMILIHAN FILE CITRA
clear; clc;
%%
% Membaca/input citra mentah dan vessel segmentasi (pilih file)
[fnameRAW, pnameRAW] = uigetfile('*.JPG','Please select RAW IMAGE(s)', 'MultiSelect', 'on');
fullnameRAW = strcat(pnameRAW,fnameRAW);n_RAW = length(fullnameRAW);
%% Khusus AVRDB
[fnameVES, pnameVES] = uigetfile({'*.jpg'},'Please select VESSEL IMAGE(s)', 'MultiSelect', 'on');
fullnameVES = strcat(pnameVES,fnameVES);
%% Khusus INSPIRE
[fnameVES, pnameVES] = uigetfile({'*.tif'},'Please select VESSEL IMAGE(s)', 'MultiSelect', 'on');
fullnameVES = strcat(pnameVES,fnameVES);
%% 
[fnameMASK, pnameMASK] = uigetfile({'*.tif'},'Please select MASK IMAGE(s)', 'MultiSelect', 'on');
fullnameMASK = strcat(pnameMASK,fnameMASK);
%% 
[fnameLES, pnameLES] = uigetfile({'*.gif'},'Please select RED LESION IMAGE(s)', 'MultiSelect', 'on');
fullnameLES = strcat(pnameLES,fnameLES);
%% 
[fnameROI, pnameROI] = uigetfile({'*.mat'},'Please select ROI file(s)', 'MultiSelect', 'on');
fullnameROI = strcat(pnameROI,fnameROI);
%% MAT FILE
[fnameSEG, pnameSEG] = uigetfile('*.mat','Please select MAT file(s)', 'MultiSelect', 'on');
fullnameSEG = strcat(pnameSEG,fnameSEG);
%% TIFF FILE
[fnameSEG, pnameSEG] = uigetfile('*.tiff','Please select VESSEL SEGMENTATION file(s)', 'MultiSelect', 'on');
fullnameSEG = strcat(pnameSEG,fnameSEG);
%% MAT kNN FILE
% [fnameKNN, pnameKNN] = uigetfile('*.mat','Please select MAT file(s)', 'MultiSelect', 'on');
% fullnameKNN = strcat(pnameKNN,fnameKNN);

%% AVRDB KOREKSI EKSU - EKSTRAKSI FITUR
DataPixel = []; DataPerim =[]; pixel=[]; DataPixelLesi=[];
for h=12:12
    tic
    % Membaca/input citra mentah dan ekstraksi kanal RGB
    % improvemnet with HSV sebelum dikonversi ke LAB (luminosity) mungkin
    % membantu (?)
    rgbImage = imread(cell2mat(fullnameRAW(h))); 
    LAB = rgb2lab(rgbImage); L = LAB(:,:,1)/100; L = adapthisteq(L,'ClipLimit',0.005); %0.005
    LAB(:,:,1) = L*100; ClaheRGB = lab2rgb(LAB);
    [counts_RGB,binLocations_RGB] = imhist(rgbImage);
%     imshowpair(rgbImage,J,'montage')
    RGB_R = rgbImage(:,:,1);
    RGB_G = rgbImage(:,:,2); 
    RGB_B = rgbImage(:,:,3);
    [row col] = size(RGB_G);
    [counts,binLocations] = imhist(RGB_G);
    B = 125/find(max(counts(50:end))==counts); RGB_G_Norm = RGB_G*B(1); 
    
%     vessImage = imread(cell2mat(fullnameVES(h)));
%     vessImage = rgb2gray(vessImage);
%     T_vI = graythresh(vessImage);
%     vI = imbinarize(vessImage, T_vI);
%     vI = 1-vI;

    Cliplimit = 0.012;
    Clahe = adapthisteq(RGB_G_Norm,'ClipLimit',Cliplimit); % CLAHE RGB_G_Norm
%     figure, imshow(Clahe);
    tic
% Gabor Wavelet
    se = strel('disk',5);
    afterOpening = imopen(Clahe, se);
%     figure, imshow(afterOpening);

    invertedimage = 255 - afterOpening;

    se = strel('disk',15);
    tophatFiltered = imtophat(invertedimage,se);
%     figure, imshow(tophatFiltered);

    kernel=[1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1;]/100;
    K=uint8(conv2(tophatFiltered,kernel,'same')); 
    J = tophatFiltered - K;

    width = 45;
    height = 45;
    kmax = pi/2;
    f = sqrt(2);
    delta =  pi/2 ;

    img_out = zeros(size(J,1), size(J,2), 10);

    for u = 0 : 9
         GW = GaborWavelet ( width, height, kmax, f, u, 4, delta ); % Create the Gabor wavelets default 3.5

    %        subplot( 1, 10, u + 1 ),imshow ( real(GW),[]);
          img_out(:,:,u+1) = imfilter(J, GW, 'symmetric');
    end

    img_out_disp = sum(abs(img_out).^2, 3).^0.45;
    img_out_disp = img_out_disp./max(img_out_disp(:));
    
    tepian = pengambilantepi(rgbImage);
    img_tepi = img_out_disp - tepian;
    T = graythresh(img_out_disp);
    thresholded = img_tepi > T;
    thresholded = thresholded/max(thresholded(:));
    bin = bwareaopen(thresholded, 35);
    toc
%     figure;
%     imshow(img_out_disp),title('hasil gabor');

% Hessian 
    num_iter = 5;   % number of iterations
    delta_t = 0.14;  % integration constant (0 <= delta_t <= 1/7)
    kappa = 3;      % gradient modulus threshold that controls the conduction
    option = 2;     % conduction coefficient functions proposed by Perona & Malik:
                    % 1 - c(x,y,t) = exp(-(nablaI/kappa).^2),
                    %     privileges high-contrast edges over low-contrast ones. 
                    % 2 - c(x,y,t) = 1./(1 + (nablaI/kappa).^2),
                    %     privileges wide regions over smaller ones.
    ad = anisodiff2D(Clahe,num_iter,delta_t,kappa,option); % Filter Difusi Anisotropik
    sigmas = 0.5:1:6.5;   % vector of scales on which the vesselness is computed 0.5;6.5
    spacing = [1;1];        % input image spacing resolution
    tau = 0.7;              % (between 0.5 and 1) lower tau -> more intense output response %default 0.7
    V = vesselness2D(ad, sigmas, spacing, tau, false); % Hessian-based
    pinggiran = imread(cell2mat(fullnameMASK(h))); 
    V2 = pinggiran.*V; V2 = V2/max(V2(:));
    T = 0.3; 
    BW = imbinarize(V2); % Thresholding
% %     figure, imshow(BW);
% 
% %     se = strel('disk',6); tophat = imtophat((255-uint8(BW)),se); 
% %     tophat = double(tophat)/double(max(tophat(:))); BW2 = imbinarize(tophat);
% % %     figure(h), imshowpair(ClaheRGB,BW,'montage')
% % %     idx = find(BW2==1); A = length(find(vI(idx)==1)); B = A*100/length(find(vI==1))
% % %     temp_name = cell2mat(fnameRAW(h));
% % %     temp_name = temp_name(1,1:length(temp_name)-4);
% % %     fout = strcat('BW_', temp_name, '.tiff');
% % %     imwrite(BW, fout, 'tiff');
% %     
    se = strel('disk',12); %default 12
    close_G = imclose(RGB_G_Norm,se); 
    exud = vesselness2D(close_G, [0.5:1:7.5], spacing, 0.65, false); % Hessian-based 0.65
    exud = pinggiran.*exud; %exud = exud/max(exud(:));
    T = 0.5; 
    BW_exud = imbinarize(exud); % Thresholding   
% % % %     figure(2), imshowpair(exud,V2,'montage')
% %     temp_name = cell2mat(fnameRAW(h));
% %     temp_name = temp_name(1,1:length(temp_name)-4);
% %     fout = strcat('Exud_', temp_name, '.tiff');
% %     imwrite(exud, fout, 'tiff');
% %     fout = strcat('BWExud_', temp_name, '.tiff');
% %     imwrite(BW_exud, fout, 'tiff');
%     
%     numWhitePixels = sum(BW_exud(:));
%     numBlackPixels = sum(~BW_exud(:));
%     DataPixel = [DataPixel; h, numWhitePixels, numBlackPixels]
% %     
% %     perimImage = bwperim(BW_exud);
% %     numPerimPixels = sum(perimImage(:));
% %     DataPerim = [DataPerim; h, numPerimPixels]
% %     
% %     area = bwarea(BW_exud);
% % %     
%     lesi = imread(cell2mat(fullnameLES(h))); % imshow(lesi)
%     numWhitePixelsLesi = sum(lesi(:));
%     numBlackPixelsLesi = sum(~lesi(:));
%     DataPixelLesi = [DataPixelLesi; h, numWhitePixelsLesi, numBlackPixelsLesi]
% % %     toc
%     [i, j] = find(BW>0); 
%     tabel = zeros(length(i),18);
%     for x=1:length(i)
%         window = double(exud(i(x)-2:i(x)+2,j(x)-2:j(x)+2));
%         window2 = double(V2(i(x)-2:i(x)+2,j(x)-2:j(x)+2));
%         window3 = double(lesi(i(x)-2:i(x)+2,j(x)-2:j(x)+2));
%         window4 = double(RGB_G_Norm(i(x)-2:i(x)+2,j(x)-2:j(x)+2));
%         window5 = double(bin(i(x)-2:i(x)+2,j(x)-2:j(x)+2));
%         tabel(x,1) = i(x);                      tabel(x,2) = j(x);
%         tabel(x,3) = double(V2(i(x),j(x)));     tabel(x,4) = mean(window2(:));  tabel(x,5) = std(window2(:));
%         tabel(x,6) = double(exud(i(x),j(x)));   tabel(x,7) = mean(window(:));   tabel(x,8) = std(window(:));
%         tabel(x,9) = double(RGB_G_Norm(i(x),j(x)))/255;     tabel(x,10) = mean(window4(:))/255;    tabel(x,11) = std(window4(:))/255; 
% %         tabel(x,9) = double(img_out_disp(i(x),j(x)));   tabel(x,10) = mean(window5(:));   tabel(x,11) = std(window5(:));
%         tabel(x,12) = double(lesi(i(x),j(x)));   tabel(x,13) = mean(window3(:));   tabel(x,14) = std(window3(:));
%         tabel(x,15) = double(thresholded(i(x),j(x)));   tabel(x,16) = mean(window5(:));   tabel(x,17) = std(window5(:));
%         tabel(x,18) = vI(i(x),j(x));   
%     end
%    
%     temp_name = cell2mat(fnameRAW(h));
%     temp_name = temp_name(1,1:length(temp_name)-4);
%     fout = strcat('EktraksiFitur_', temp_name, '.mat');
%     save(fout,'tabel');
%     toc
% %     
%     A = edge(BW); [x y] = find(A==1);
%     figure(h), imshow(ClaheRGB), hold on, scatter(y,x,3,'filled','s','green'), hold off
end

%% AVRDB SEGMENTASI
tic
Data = []; fitur = [3:17]; Parameter = []; %fitur = [3:17]
load('E:\Image AVRDB kelas\image\OD kanan\KNN_5_exud_30.mat');
    
for h=25:25
%     % Membaca/input citra mentah dan ekstraksi kanal RGB
    tic
    rgbImage = imread(cell2mat(fullnameRAW(h))); 
    LAB = rgb2lab(rgbImage); L = LAB(:,:,1)/100; L = adapthisteq(L,'ClipLimit',0.005); 
    LAB(:,:,1) = L*100; ClaheRGB = lab2rgb(LAB); 
    RGB_R = rgbImage(:,:,1);
    RGB_G = rgbImage(:,:,2); 
    RGB_B = rgbImage(:,:,3);
    [row col] = size(RGB_G);
    [counts,binLocations] = imhist(RGB_G);
    B = 125/find(max(counts(50:end))==counts); RGB_G_Norm = RGB_G*B(1); 

    load(cell2mat(fullnameSEG(h)));
%     load(cell2mat(fullnameKNN(h)));
    X = tabel(:,fitur);
    [Y, Y_score] = predict(Mdl,X);
    BW2 = zeros(row,col); BW = BW2; %bin = BW2; %bin2 =zeros(row,col); 
    for i=1:size(tabel,1)
        BW2(tabel(i,1),tabel(i,2)) = Y(i);
        BW(tabel(i,1),tabel(i,2)) = 1;
    end
    Morph = bwmorph(BW2,'clean',Inf);
%     figure, imshow(Morph); title('Clean');
    Morph = bwmorph(Morph,'fill',Inf);
%     figure, imshow(Morph); title('Fill');
    Morph = bwmorph(Morph,'bridge',Inf);
%     figure, imshow(Morph); title('Bridge');
    Morph = bwmorph(Morph,'diag',Inf);
%     figure, imshow(Morph); title('diag');
    Morph = bwmorph(Morph,'close',1);
%     figure, imshow(Morph); title('close');
%     figure(h), imshowpair(ClaheRGB,Morph,'montage')
    
    temp_name = cell2mat(fnameSEG(h));
    temp_name = temp_name(1,14:length(temp_name)-4);
    fout = strcat('Ves_KNN_5_exud_30', temp_name, '.tiff');
    imwrite(Morph, fout, 'tiff');
    
%     Morph = imread(cell2mat(fullnameSEG(h)));
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
    
%     Data = [Data; fnameSEG(h), Acc, Sen, Spec, Prec];
%     fout = strcat('Performa Segmentasi dengan Gabor', '.mat');
%     save(fout, 'Data');
    toc
%     figure(h), imshowpair(ClaheRGB,BW,'montage')
%     A = edge(BW2); [x y] = find(A==1);
%     figure(h), imshow(ClaheRGB), hold on, scatter(y,x,3,'filled','s','green'), hold off
end
toc
%% INSPIRE KOREKSI EKSU - EKSTRAKSI FITUR
for h=11:40
    % Membaca/input citra mentah dan ekstraksi kanal RGB
    rgbImage = imread(cell2mat(fullnameRAW(h))); 
    LAB = rgb2lab(rgbImage); L = LAB(:,:,1)/100; L = adapthisteq(L,'ClipLimit',0.005); 
    LAB(:,:,1) = L*100; ClaheRGB = lab2rgb(LAB); 
%     imshowpair(rgbImage,J,'montage')
    RGB_R = rgbImage(:,:,1);
    RGB_G = rgbImage(:,:,2); 
    RGB_B = rgbImage(:,:,3);
    [row col] = size(RGB_G);
    [counts,binLocations] = imhist(RGB_G);
    B = 125/find(max(counts(50:end))==counts); RGB_G_Norm = RGB_G*B(1);  

%     % Pre-processing dan Segmentation
    Cliplimit = 0.012;
    Clahe = adapthisteq(RGB_G_Norm,'ClipLimit',Cliplimit); % CLAHE
    num_iter = 5;   % number of iterations
    delta_t = 0.14;  % integration constant (0 <= delta_t <= 1/7)
    kappa = 3;      % gradient modulus threshold that controls the conduction
    option = 2;     % conduction coefficient functions proposed by Perona & Malik:
                    % 1 - c(x,y,t) = exp(-(nablaI/kappa).^2),
                    %     privileges high-contrast edges over low-contrast ones. 
                    % 2 - c(x,y,t) = 1./(1 + (nablaI/kappa).^2),
                    %     privileges wide regions over smaller ones.
    ad = anisodiff2D(Clahe,num_iter,delta_t,kappa,option); % Filter Difusi Anisotropik
    sigmas = 0.5:1:9.5;   % vector of scales on which the vesselness is computed
    spacing = [1;1];        % input image spacing resolution
    tau = 1;              % (between 0.5 and 1) lower tau -> more intense output response
    V = vesselness2D(ad, sigmas, spacing, tau, false); % Hessian-based
%     masking= MaskAVRDB(V); %tambahanku sendiri ya ini
    pinggiran = imread(cell2mat(fullnameMASK(h))); 
    V2 = pinggiran.*V; V2 = V2/max(V2(:));
    T = 0.4;
    BW = imbinarize(V2); % Thresholding
%     imshowpair(RGB_G_Norm,BW,'montage')
%     BW2 = BW-imbinarize(V2, 0.8); % Thresholding
    figure(1), imshowpair(BW,V2,'montage')
    
%     se = strel('disk',30);
%     close_G = imclose(RGB_G_Norm,se);
%     exud = vesselness2D(close_G, [0.5:1:10.5], spacing, 0.65, false); % Hessian-based
%     exud = pinggiran.*exud; exud = exud/max(exud(:));
%     T = 0.5;
%     BW_exud = imbinarize(exud,T); % Thresholding   
%     BW_exud2 = BW2+BW_exud;
%     figure(h), imshowpair(exud,BW_exud,'montage')

%     lesi = imread(cell2mat(fullnameLES(h))); % imshow(lesi)
%     [i, j] = find(BW>0); 
%     tabel = zeros(length(i),14);
%     for x=1:length(i)
%         window = double(exud(i(x)-2:i(x)+2,j(x)-2:j(x)+2));
%         window2 = double(V2(i(x)-2:i(x)+2,j(x)-2:j(x)+2));
%         window3 = double(lesi(i(x)-2:i(x)+2,j(x)-2:j(x)+2));
%         window4 = double(RGB_G_Norm(i(x)-2:i(x)+2,j(x)-2:j(x)+2));
%         tabel(x,1) = i(x);                      tabel(x,2) = j(x);
%         tabel(x,3) = double(V2(i(x),j(x)));     tabel(x,4) = mean(window2(:));  tabel(x,5) = std(window2(:));
%         tabel(x,6) = double(exud(i(x),j(x)));   tabel(x,7) = mean(window(:));   tabel(x,8) = std(window(:));
%         tabel(x,9) = double(RGB_G_Norm(i(x),j(x)))/255;     tabel(x,10) = mean(window4(:))/255; 
%         tabel(x,11) = std(window4(:))/255;      
%         tabel(x,12) = double(lesi(i(x),j(x)));   tabel(x,13) = mean(window3(:));   tabel(x,14) = std(window3(:));           
%     end
    
    temp_name = cell2mat(fnameRAW(h));
    temp_name = temp_name(1,1:length(temp_name)-4);
%     fout = strcat('KoreksiEXU_', temp_name, '.mat');
%     save(fout,'tabel');
    fout = strcat('Ves_', temp_name, '.tiff');
    imwrite(BW, fout, 'tiff');
    
%     figure(h), imshowpair(V2,exud,'montage')

end

%% INSPIRE SEGMENTASI
Data = []; fitur = [3:14];
% load('D:\TA Yuni\All Code\Code Not Resized\Model Segmentasi\Model_finKNN_12.mat');
for h=2:10
    % Membaca/input citra mentah dan ekstraksi kanal RGB
    rgbImage = imread(cell2mat(fullnameRAW(h))); 
    LAB = rgb2lab(rgbImage); L = LAB(:,:,1)/100; L = adapthisteq(L,'ClipLimit',0.005); 
    LAB(:,:,1) = L*100; ClaheRGB = lab2rgb(LAB); 
%     imshowpair(rgbImage,J,'montage')
    RGB_R = rgbImage(:,:,1);
    RGB_G = rgbImage(:,:,2); 
    RGB_B = rgbImage(:,:,3);
    [row col] = size(RGB_G);
    [counts,binLocations] = imhist(RGB_G);
    B = 125/find(max(counts(50:end))==counts); RGB_G_Norm = RGB_G*B(1); 

    load(cell2mat(fullnameSEG(h)));
    load(cell2mat(fullnamekNN(h)));
    X = tabel(:,3:14);
    [Y, Y_score] = predict(Mdl,X);
    BW2 = zeros(row,col); BW = BW2;
    for i=1:size(tabel,1)
        BW2(tabel(i,1),tabel(i,2)) = Y(i);
        BW(tabel(i,1),tabel(i,2)) = 1;
    end
    Morph = bwmorph(BW2,'clean',Inf);
    Morph = bwmorph(Morph,'fill',Inf);
    Morph = bwmorph(Morph,'bridge',Inf);
    Morph = bwmorph(Morph,'diag',Inf);
    Morph = bwmorph(Morph,'close',1);
    imshowpair(BW,BW2,'montage')
    
    temp_name = cell2mat(fnameSEG(h));
    temp_name = temp_name(1,12:length(temp_name)-4);
    fout = strcat('Ves_', temp_name, '.tiff');
    imwrite(BW2, fout, 'tiff');

%     A = edge(BW2); [x y] = find(A==1);
%     figure(h), imshow(ClaheRGB), hold on, scatter(y,x,3,'filled','s','green'), hold off
end

%% INSPIRE PERFORMA SEG
fitur = [3:11]; Data = [];
for h=1:1
    rgbImage = imread(cell2mat(fullnameRAW(h)));
    vessImage = imread(cell2mat(fullnameVES(h)));
    vR = vessImage(:,:,1); i=find(vR~=0); vR(i) = 255;
    vG = vessImage(:,:,2); i=find(vG~=0); vG(i) = 255;
    vB = vessImage(:,:,3); i=find(vB~=0); vB(i) = 255;
    vRGB = cat(3, vR, vG, vB);
    vGT = logical(vR+vG+vB);
    h
    [row col color] = size(rgbImage);       % INSPIRE
    Morph = imread(cell2mat(fullnameSEG(h)));
%     Morph = bwmorph(Morph,'clean',Inf);
%     Morph = bwmorph(Morph,'fill',Inf);
%     Morph = bwmorph(Morph,'bridge',Inf);
%     Morph = bwmorph(Morph,'diag',Inf);
%     Morph = bwmorph(Morph,'close',1);
%     multi = cat(3, BW_Ves, Ves, Ves2);
%     figure(h), montage(multi,'size',[1 3])
    
    Skel = bwmorph(Morph,'skel',Inf); 
    Skel2 = bwmorph(Skel,'spur',60); 
    [S] = DivideSkel(Skel2); n_seg = max(S(:,3));
    for i=1:n_seg
        idx = find(S(:,3)==i);
        if length(idx)<3
            for j=1:length(idx)
                Skel2(S(idx(j),1),S(idx(j),2)) = 0;
            end
        end
    end
%     imshowpair(vGT, Skel2, 'montage'), title ('Centerline GT (kiri) & Centerline Segmentasi (kanan)'); 
    vGT2 = bwmorph(vGT,'skel',3); 
%     imshowpair(vGT2, Skel2, 'montage'), , title ('Centerline GT (kiri) & Centerline Segmentasi (kanan)'); 

%     idx = find(vGT>0); A = find(Morph(idx)>0);
%     Data = [Data; h length(A)/length(idx)*100];
    
%     vGT2 = vGT;
%     TP = 0; FP = 0; FN = 0;
%     fTP = zeros(row, col); fFP = fTP; fFN = fTP;
%     for i=1:row
%         for j=1:col
%             if vGT2(i,j)==1 && Skel2(i,j)==1
%                 TP = TP+1; fTP(i,j) = 1;
%             elseif vGT2(i,j)==1 && Skel2(i,j)==0
%                 FN = FN+1; fFN(i,j) = 1;
%             elseif vGT2(i,j)==0 && Skel2(i,j)==1
%                 FP = FP+1; fFP(i,j) = 1;
%             end
%         end
%     end
%     allGT = length(find(vGT2==1)); allSEG = length(find(Skel2==1));
%     pGT = TP*100/allGT; pSEG = TP*100/allSEG;
%     Data = [Data; h TP FP FN allSEG allGT pSEG pGT];
    
    vGT2 = vGT;
    TP_GT = 0; TP_SEG = 0; FP = 0; FN = 0;
    fTP_GT = zeros(row, col); fFP = fTP_GT; fFN = fTP_GT; fTP_SEG = fTP_GT;
    for i=3:row-2
        for j=3:col-2
            wGT = vGT2(i-2:i+2,j-2:j+2); wGT = sum(wGT(:));
            wSEG = Skel2(i-2:i+2,j-2:j+2); wSEG = sum(wSEG(:));
            if wGT>0 && Skel2(i,j)==1
                TP_SEG = TP_SEG+1; fTP_SEG(i,j) = 1;
            elseif wSEG>0 && vGT2(i,j)==1
                TP_GT = TP_GT+1; fTP_GT(i,j) = 1;
            elseif vGT2(i,j)==1 && wSEG==0
                FN = FN+1; fFN(i,j) = 1;
            elseif wGT==0 && Skel2(i,j)==1
                FP = FP+1; fFP(i,j) = 1;
            end
        end
    end
    allGT = length(find(vGT2==1)); allSEG = length(find(Skel2==1));
    pGT = TP_GT*100/allGT; pSEG = TP_SEG*100/allSEG;
    Data = [Data; h TP_SEG TP_GT FP FN allSEG allGT pSEG pGT];
    
    figure(h), t = tiledlayout(1,3);
    nexttile, imshow(rgbImage), title ('Citra Asli'),
    nexttile, imshow(fTP_SEG), title ('True Positive'), 
    nexttile, imshow(fFP), title ('False Positive'),
    nexttile, imshowpair(fTP_SEG,fFP), title ('Centerline Segmentasi'),
%     t.TileSpacing = 'none'
%     t.Padding = 'compact'
%     
%     figure(h), t = tiledlayout(1,3);
    nexttile, imshow(rgbImage), title ('Citra Asli'),
    nexttile, imshow(fTP_GT), title ('True Positive'), 
    nexttile, imshow(fFN), title ('False Negative'),
    nexttile, imshowpair(fTP_GT,fFN), title ('Centerline Ground Truth'),
    t.TileSpacing = 'none'
    t.Padding = 'compact'
end

%% INSPIRE KNUDTSON
dataAVR = []; dataArt = []; dataVein = [];
for h=1:1
    h
    % Membaca/input citra mentah dan ekstraksi kanal RGB
    rgbImage = imread(cell2mat(fullnameRAW(h))); 
    RGB_R = rgbImage(:,:,1);
    RGB_G = rgbImage(:,:,2); 
    RGB_B = rgbImage(:,:,3);
    [row col] = size(RGB_G);

    Morph = imread(cell2mat(fullnameSEG(h)));
%     Morph = bwmorph(Morph,'clean',Inf);
%     Morph = bwmorph(Morph,'fill',Inf);
%     Morph = bwmorph(Morph,'bridge',Inf);
%     Morph = bwmorph(Morph,'diag',Inf);
%     Morph = bwmorph(Morph,'close',1);
    BW = Morph; % Thresholding

    vessImage = imread(cell2mat(fullnameVES(h)));
    vR = vessImage(:,:,1); i=find(vR~=0); vR(i) = 255;
    vG = vessImage(:,:,2); i=find(vG~=0); vG(i) = 255;
    vB = vessImage(:,:,3); i=find(vB~=0); vB(i) = 255;
    vRGB = cat(3, vR, vG, vB);
    vR = logical(vR-vG); vB = logical(vB-vG); 
%     figure(h), imshow(BW), hold on, imshow(vRGB), hold off
    
    BW_Ves = BW;
    [row col] = size(BW_Ves);
    
    OD = load(cell2mat(fullnameROI(h)));
    ROI = zeros(row,col); ROI = logical(ROI);
    for i=1:row
        for j=1:col
            d_OD = sqrt((i-OD.X)^2+(j-OD.Y)^2);
            if (d_OD > 1.5*OD.radius) && (d_OD < 3*OD.radius)
                ROI(i,j) = 1;
            end
        end
    end
    
    T = 30;
%     figure(h), imshow(BW), hold on,
    [Segmen] = DivideSkel(vR.*ROI.*BW_Ves); Art = []; arr_D = []; seg = 1; arr_T = [];
    for i=1:length(Segmen)
        [grad D] = Calculate_Diameter(BW_Ves, Segmen(i,1), Segmen(i,2));
        if (Segmen(i,2)>=OD.Y)
            theta_OD = atan((Segmen(i,1)-OD.X)/(Segmen(i,2)-OD.Y));
            theta_OD = mod(-theta_OD,2*pi);
        else
            theta_OD = atan((Segmen(i,1)-OD.X)/(Segmen(i,2)-OD.Y));
            theta_OD = -theta_OD + pi;
        end
        Segmen(i,4) = D;
%         x1 = Segmen(i,1); y1 = Segmen(i,2); tepi = 0; x=[]; y=[];
%         x = [x; x1]; y = [y; y1];
%         while not(tepi)
%             if abs(grad)>1
%                 y1 = y1+1;
%                 x1 = round((y1-Segmen(i,2))/grad+Segmen(i,1));
%             else
%                 x1 = x1+1;
%                 y1 = round(grad*(x1-Segmen(i,1))+Segmen(i,2));
%             end
%             if (BW(x1,y1)==0)
%                 tepi = 1;
%             else
%                 x = [x; x1]; y = [y; y1];
%             end
%         end
%         x1 = Segmen(i,1); y1 = Segmen(i,2); tepi = 0;
%         while not(tepi)
%             if abs(grad)>1
%                 y1 = y1-1;
%                 x1 = round((y1-Segmen(i,2))/grad+Segmen(i,1));
%             else
%                 x1 = x1-1;
%                 y1 = round(grad*(x1-Segmen(i,1))+Segmen(i,2));
%             end
%             if (BW(x1,y1)==0)
%                 tepi = 1;
%             else
%                 x = [x; x1]; y = [y; y1];
%             end
%         end
%         scatter(y,x,1,'filled');
        if (seg==Segmen(i,3))
            arr_D = [arr_D, D]; arr_T = [arr_T, theta_OD];
        else
            if length(arr_D) > T
                Art = [Art; h seg mean(arr_D) std(arr_D) min(arr_D) median(arr_D) mode(arr_D) max(arr_D) mean(arr_T)];
            end
            seg = Segmen(i,3); arr_D = []; arr_T = [];
            arr_D = [arr_D, D]; arr_T = [arr_T, theta_OD];
        end
        if (i==length(Segmen))
            if length(arr_D) > T
                Art = [Art; h seg mean(arr_D) std(arr_D) min(arr_D) median(arr_D) mode(arr_D) max(arr_D) mean(arr_T)];
            end
        end
    end
    segArt = Segmen;
    
    [Segmen] = DivideSkel(vB.*ROI.*BW_Ves); Vein = []; arr_D = []; seg = 1; arr_T = [];
    for i=1:length(Segmen)
        [grad D] = Calculate_Diameter(BW_Ves, Segmen(i,1), Segmen(i,2));
        Segmen(i,4) = D;
        if (Segmen(i,2)>=OD.Y)
            theta_OD = atan((Segmen(i,1)-OD.X)/(Segmen(i,2)-OD.Y));
            theta_OD = mod(-theta_OD,2*pi);
        else
            theta_OD = atan((Segmen(i,1)-OD.X)/(Segmen(i,2)-OD.Y));
            theta_OD = -theta_OD + pi;
        end
        if (seg==Segmen(i,3))
            arr_D = [arr_D, D]; arr_T = [arr_T, theta_OD];
        else
            if length(arr_D) > T
                Vein = [Vein; h seg mean(arr_D) std(arr_D) min(arr_D) median(arr_D) mode(arr_D) max(arr_D) mean(arr_T)];
            end
            seg = Segmen(i,3); arr_D = []; arr_T = [];
            arr_D = [arr_D, D]; arr_T = [arr_T, theta_OD];
        end
        if (i==length(Segmen))
            if length(arr_D) > T
                Vein = [Vein; h seg mean(arr_D) std(arr_D) min(arr_D) median(arr_D) mode(arr_D) max(arr_D) mean(arr_T)];
            end
        end
    end
    segVein = Segmen;
%     hold off

    idx = 3;
    Art = sortrows(Art, idx, 'descend'); 
    Vein = sortrows(Vein, idx, 'descend');
    i=1;
    while i<size(Art,1)
        difD = abs(Art(i,3)-Art(i+1,3)); difT = abs(Art(i,9)-Art(i+1,9));
        if difD<=1.5 && difT<=0.2
            Art(i+1,:) = [];
        else
            i=i+1;
        end
    end
    i=1;
    while i<size(Vein,1)
        difD = abs(Vein(i,3)-Vein(i+1,3)); difT = abs(Vein(i,9)-Vein(i+1,9));
        if difD<=1.5 && difT<=0.2
            Vein(i+1,:) = [];
        else
            i=i+1;
        end
    end
    
    [N, CRAE, CRVE, AVR] = calAVR(Art(:,idx),Vein(:,idx));
    dataArt = [dataArt; Art(1:N,:)]; dataVein = [dataVein; Vein(1:N,:)]; 
    dataAVR = [dataAVR; h, N, CRAE, CRVE, AVR];
end

%% INSPIRE PARR HUBBARD
dataAVR = []; dataArt = []; dataVein = [];
for h=1:40
    % Membaca/input citra mentah dan ekstraksi kanal RGB
    rgbImage = imread(cell2mat(fullnameRAW(h))); 
    RGB_R = rgbImage(:,:,1);
    RGB_G = rgbImage(:,:,2); 
    RGB_B = rgbImage(:,:,3);
    [row col] = size(RGB_G);

    Morph = imread(cell2mat(fullnameSEG(h)));
    Morph = bwmorph(Morph,'clean',Inf);
    Morph = bwmorph(Morph,'fill',Inf);
    Morph = bwmorph(Morph,'bridge',Inf);
    Morph = bwmorph(Morph,'diag',Inf);
    Morph = bwmorph(Morph,'close',1);
    BW = Morph; % Thresholding

    vessImage = imread(cell2mat(fullnameVES(h)));
    vR = vessImage(:,:,1); i=find(vR~=0); vR(i) = 255;
    vG = vessImage(:,:,2); i=find(vG~=0); vG(i) = 255;
    vB = vessImage(:,:,3); i=find(vB~=0); vB(i) = 255;
    vRGB = cat(3, vR, vG, vB);
    vR = logical(vR-vG); vB = logical(vB-vG); 
%     figure(h), imshow(BW), hold on, imshow(vRGB), hold off
    
    BW_Ves = BW;
    [row col] = size(BW_Ves);
    
    OD = load(cell2mat(fullnameROI(h)));
    ROI = zeros(row,col); ROI = logical(ROI);
    for i=1:row
        for j=1:col
            d_OD = sqrt((i-OD.X)^2+(j-OD.Y)^2);
            if (d_OD > 1.5*OD.radius) && (d_OD < 3*OD.radius)
                ROI(i,j) = 1;
            end
        end
    end
    
%     figure(h), imshow(BW), hold on,
    [Segmen] = DivideSkel(vR.*ROI.*BW_Ves); Art = []; arr_D = []; seg = 1;
    for i=1:length(Segmen)
        [grad D] = Calculate_Diameter(BW_Ves, Segmen(i,1), Segmen(i,2));
        Segmen(i,4) = D;

        if (seg==Segmen(i,3))
            arr_D = [arr_D, D];
        else
            Art = [Art; h seg mean(arr_D) std(arr_D) min(arr_D) median(arr_D) mode(arr_D) max(arr_D)];
            seg = Segmen(i,3); arr_D = []; arr_D = [arr_D, D];
        end
        if (i==length(Segmen))
            Art = [Art; h seg mean(arr_D) std(arr_D) min(arr_D) median(arr_D) mode(arr_D) max(arr_D)];
        end
    end
    
    [Segmen] = DivideSkel(vB.*ROI.*BW_Ves); Vein = []; arr_D = []; seg = 1;
    for i=1:length(Segmen)
        [grad D] = Calculate_Diameter(BW_Ves, Segmen(i,1), Segmen(i,2));
        Segmen(i,4) = D;
        if (seg==Segmen(i,3))
            arr_D = [arr_D, D];
        else
            Vein = [Vein; h seg mean(arr_D) std(arr_D) min(arr_D) median(arr_D) mode(arr_D) max(arr_D)];
            seg = Segmen(i,3); arr_D = []; arr_D = [arr_D, D];
        end
        if (i==length(Segmen))
            Vein = [Vein; h seg mean(arr_D) std(arr_D) min(arr_D) median(arr_D) mode(arr_D) max(arr_D)];
        end
    end
%     hold off

    Art = sortrows(Art, 3, 'descend'); 
    Vein = sortrows(Vein, 3, 'descend');
    idx = find(Art(:,3)<10); Art(idx,:) = [];
    idx = find(Vein(:,3)<10); Vein(idx,:) = [];
    
    N = min([size(Vein,1),size(Art,1)]); N = round(N/2);
    CRAE = calCRAE(Art(1:N,3)); CRVE = calCRVE(Vein(1:N,3)); AVR = CRAE/CRVE;
%     dataArt = [dataArt; Art(1:N,:)]; dataVein = [dataVein; Vein(1:N,:)]; 
    dataAVR = [dataAVR; h, CRAE, CRVE, AVR];
    h
    

end
%% INSPIRE PARR HUBBARD 2
dataAVR = []; dataArt = []; dataVein = [];
for h=1:40
    % Membaca/input citra mentah dan ekstraksi kanal RGB
    rgbImage = imread(cell2mat(fullnameRAW(h))); 
    RGB_R = rgbImage(:,:,1);
    RGB_G = rgbImage(:,:,2); 
    RGB_B = rgbImage(:,:,3);
    [row col] = size(RGB_G);

    Morph = imread(cell2mat(fullnameSEG(h)));
    BW = Morph; % Thresholding

    vessImage = imread(cell2mat(fullnameVES(h)));
    vR = vessImage(:,:,1); i=find(vR~=0); vR(i) = 255;
    vG = vessImage(:,:,2); i=find(vG~=0); vG(i) = 255;
    vB = vessImage(:,:,3); i=find(vB~=0); vB(i) = 255;
    vRGB = cat(3, vR, vG, vB);
    vR = logical(vR-vG); vB = logical(vB-vG); 
%     figure(h), imshow(BW), hold on, imshow(vRGB), hold off
    
    BW_Ves = BW;
    [row col] = size(BW_Ves);
    
    OD = load(cell2mat(fullnameROI(h)));
    ROI = zeros(row,col); ROI = logical(ROI); s_ROI = ROI; l_ROI = ROI;
    for i=1:row
        for j=1:col
            d_OD = sqrt((i-OD.X)^2+(j-OD.Y)^2);
            if (d_OD > 1.5*OD.radius) && (d_OD < 3*OD.radius)
                ROI(i,j) = 1;
            end
            if (d_OD > 1.5*OD.radius) && (d_OD < 1.55*OD.radius)
                l_ROI(i,j) = 1;
            end
            if (d_OD > 2.95*OD.radius) && (d_OD < 3*OD.radius)
                s_ROI(i,j) = 1;
            end
        end
    end
    
    Art = [];
    A = s_ROI.*vR.*BW_Ves; [x y] = find(A==1);
    for i=1:length(x)
        [grad D] = Calculate_Diameter(BW_Ves, x(i), y(i));
        Art(i,1) = D; 
    end
    A = l_ROI.*vR.*BW_Ves; [x, y] = find(A==1);
    for i=1:length(x)
        [grad, D] = Calculate_Diameter(BW_Ves, x(i), y(i));
        Art(i,2) = D; 
    end
    temp = sort(Art(:,1),'descend'); Art(:,1) = temp;
    temp = sort(Art(:,2),'descend'); Art(:,2) = temp;
    meanS = mean(Art(1:10,1)); meanL = mean(Art(1:10,2));
    CRAE = calCRAE([meanL, meanS]);
    
    Vein = [];
    V = s_ROI.*vB.*BW_Ves; [x y] = find(V==1);
    for i=1:length(x)
        [grad D] = Calculate_Diameter(BW_Ves, x(i), y(i));
        Vein(i,1) = D; 
    end
    V = l_ROI.*vB.*BW_Ves; [x y] = find(V==1);
    for i=1:length(x)
        [grad D] = Calculate_Diameter(BW_Ves, x(i), y(i));
        Vein(i,2) = D; 
    end
    temp = sort(Vein(:,1),'descend'); Vein(:,1) = temp;
    temp = sort(Vein(:,2),'descend'); Vein(:,2) = temp;
    meanS = mean(Vein(1:10,1)); meanL = mean(Vein(1:10,2));
    CRVE = calCRVE([meanL, meanS]);
    h
    AVR = CRAE/CRVE; dataAVR = [dataAVR; h, CRAE, CRVE, AVR];

end