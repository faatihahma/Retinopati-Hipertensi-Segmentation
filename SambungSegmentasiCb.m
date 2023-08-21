%% Sambung Segmentasi

% Membaca/input citra mentah dan vessel segmentasi (pilih file)
[fnameRAW, pnameRAW] = uigetfile('*.JPG','Please select RAW IMAGE(s)', 'MultiSelect', 'on');
fullnameRAW = strcat(pnameRAW,fnameRAW);n_RAW = length(fullnameRAW);
%% Khusus AVRDB
[fnameVES, pnameVES] = uigetfile({'*.jpg'},'Please select VESSEL IMAGE(s)', 'MultiSelect', 'on');
fullnameVES = strcat(pnameVES,fnameVES);
%% TIFF FILE
[fnameSEG, pnameSEG] = uigetfile('*.tiff','Please select VESSEL SEGMENTATION file(s)', 'MultiSelect', 'on'); %*.tiff
fullnameSEG = strcat(pnameSEG,fnameSEG);
%%
% Baca citra GT ARTERY & VEIN
[fnameARTERI, pnameARTERI] = uigetfile('*.jpg','Please select ARTERY IMAGE(s)', 'MultiSelect', 'on');
fullnameARTERI = strcat(pnameARTERI,fnameARTERI);
%%
[fnameVENA, pnameVENA] = uigetfile('*.jpg','Please select VEIN IMAGE(s)', 'MultiSelect', 'on');
fullnameVENA = strcat(pnameVENA,fnameVENA);
%%
% Baca file ROI
[fnameROI, pnameROI] = uigetfile({'*.mat'},'Please select ROI file(s)', 'MultiSelect', 'on');
fullnameROI = strcat(pnameROI,fnameROI);
%%
% SKELETON Telea
[fnameSKEL, pnameSKEL] = uigetfile('*.png','Select SKELETON Telea image(s)', 'MultiSelect', 'on');
fullnameSKEL = strcat(pnameSKEL,fnameSKEL);
%%
% SKELETON GT
[fnameSKEL_GT, pnameSKEL_GT] = uigetfile('*.png','Select SKELETON Telea image(s)', 'MultiSelect', 'on');
fullnameSKEL_GT = strcat(pnameSKEL_GT,fnameSKEL_GT);
%%
All_data = [];
for h = 1:1
    rgbImage = imread(cell2mat(fullnameRAW(h))); 
    RGB_R = rgbImage(:,:,1);
    RGB_G = rgbImage(:,:,2); 
    RGB_B = rgbImage(:,:,3);
    RGB_RG = double(RGB_R).^2 + double(RGB_G).^2;
    RGB_RG = sqrt(0.5*RGB_RG);
    [row col] = size(RGB_G);
    BW = imread(cell2mat(fullnameSEG(h)));
%     BW = rgb2gray(BW);
%     T_vI = graythresh(BW);
%     vI = imbinarize(BW, T_vI);
%     BW = 1-vI;

%     vessImage = imread(cell2mat(fullnameVES(h)));
%     vessImage = rgb2gray(vessImage);
%     T_vI = graythresh(vessImage);
%     vI = imbinarize(vessImage, T_vI);
%     BW = 1-vI;
    
    % Membaca ground truth arteri-vena
    arteriImage = imread(cell2mat(fullnameARTERI(h)));
    arteriImage = arteriImage(:,:,2);
    venaImage = imread(cell2mat(fullnameVENA(h)));
    venaImage = venaImage(:,:,2);

    % Ekstraksi ROI
    OD = open(cell2mat(fullnameROI(h)));
    ROI = zeros(row,col);
    ROI = logical(ROI);
    for i=1:row
        for j=1:col
            d_OD = sqrt((i-OD.X)^2+(j-OD.Y)^2);
            if (d_OD > 1.5*OD.radius) && (d_OD < 3*OD.radius) % -------->>> GANTI ROI 
                ROI(i,j) = 1;
            end
        end
    end
    BW_final = ROI.*BW; %ganti juga kalau ngukur lainnya

    Skel = imread(cell2mat(fullnameSKEL(h)));
%     Skel = imread(cell2mat(fullnameSKEL_GT(h))); %kalau yang dihitng GT pakai Skel GT
    Skel_fin = ROI.*Skel;
    Skel_fin = bwmorph(Skel_fin, 'clean',Inf);
    BW_final_2 = BW_final;
    branch = bwmorph(Skel_fin, 'branchpoints',1);
    
    [i j] = find(branch==1); A = [];
    for k=1:length(i)
        win = Skel_fin(i(k)-2:i(k)+2,j(k)-2:j(k)+2);
        A = [A; i(k) j(k) sum(win(:))];
        if sum(win(:))<7
            branch(i(k),j(k)) = 0;
        end
        if sum(win(:))>=7
            for m = i(k)-12:i(k)+12
                for n = j(k)-12:j(k)+12
                    cond = sqrt((m-i(k))^2+(n-j(k))^2);
                    if (cond<=12)
                        BW_final_2(m,n) = 0;
                    end
                end
            end
        end
    end

    [Segmen] = DivideSkel(BW_final_2); 
    n_seg = max(Segmen(:,3)); seg = Segmen(:,3);
    segmen_x = Segmen(:,2); segmen_y = Segmen(:,1);

    f1 = figure();
    imshow (BW),
    hold on
   
    F_pixel = []; F_profSeg = []; Feature = []; GT = []; 
    for i=1:n_seg
        idx = find(seg==i);
        x = segmen_x(idx);
        y = segmen_y(idx);
%         scatter(x, y, 5, 's', 'filled');
%     end
        D_seg = []; vote_vess = [];

        for j=1:length(idx)
            if (Skel(y(j),x(j))==1)
                [grad D] = Calculate_Diameter(BW, y(j), x(j)); % BW
%                 [A] = Show_Diameter(grad, D, y(j), x(j), BW)
                D_seg = [D_seg; D];
                r_OD = sqrt((y(j)-OD.X)^2+(x(j)-OD.Y)^2)/OD.radius;
                r_Center = sqrt((y(j)-row/2)^2+(x(j)-col/2)^2);
                if (x(j)>OD.Y)
                    theta_OD = atan((y(j)-OD.X)/(x(j)-OD.Y));
                    theta_OD = mod(-theta_OD,2*pi);
                else
                    theta_OD = atan((y(j)-OD.X)/(x(j)-OD.Y));
                    theta_OD = -theta_OD + pi;
                end
            else
                grad = 0; D = 0; r_OD = 0; r_Center = 0; theta_OD = 0;
                zoneA = zeros(1,65); zoneB = zeros(1,65); zoneC = zeros(1,65);
            end

            if (arteriImage(y(j), x(j)) ~= 255)
                vote_vess = [vote_vess; 1]; % ARTERI
            elseif (venaImage(y(j), x(j)) ~= 255)
                vote_vess = [vote_vess; -1]; % VENA
            else
                vote_vess = [vote_vess; 0]; % NOT BOTH
            end

            if (D~=0)
                F_pixel = [F_pixel; i, y(j), x(j), grad, D, D/OD.radius, r_OD, theta_OD];
            end
        end
        vess = sum(vote_vess);
        if (vess>0)
            GT = [GT; repmat(cellstr('Arteri'), length(D_seg),1)]; % ARTERI
        elseif (vess<0) 
            GT = [GT; repmat(cellstr('Vena'), length(D_seg),1)]; % VENA
        else
            GT = [GT; repmat(cellstr('NB'), length(D_seg),1)]; % Not Both
        end
    end
    Feature = [F_pixel];
    % Masukin fitur ke tabel
    Table_data = array2table(Feature,...
        'VariableNames',{'Segmen','Baris','Kolom','Grad','Diameter','NormD','DistToOD','AngleToOD'});
    Table_data.Vessel = GT;
    temp_name = cell2mat(fnameRAW(h));
    temp_name = temp_name(1,1:length(temp_name)-4);
    Filename = repmat({temp_name},size(Table_data, 1),1);
    Table_data.Filename = Filename;
    Table_data = [Table_data(:,end), Table_data(:,1:end-1)];
    All_data = [All_data; Table_data];
end
% writetable(All_data,'cb_5_art_2-3.csv');
%%
% READ FILE
clear all;
% Membaca/input tabel dari csv (pilih file) 
% Feature_A/V
[fnameCSV, pnameCSV] = uigetfile('*.csv','Please select Data file(s)', 'MultiSelect', 'off');
fullnameCSV = strcat(pnameCSV,fnameCSV);
Data = readtable(fullnameCSV);
%% Pixel-based jadi Segment-based
% Segmen yg sama dihitung diameter (mean,med,dll)
[row col] = size(Data); 
numCL = 2; % INSPIRE di Thea di Yuni avrdb 2
% [row col] = size(Data); numCL = 15; % AVRDB
T = []; GT = []; Classifier = []; FN = []; 
arrD = []; arrVes = []; arrT = []; arrX = []; arrY = [];
name = Data.Filename(1); seg = Data.Segmen(1);
for i=1:row
    bool = strcmp(char(name),char(Data.Filename(i)));   % bandingin Filename ke-1 dan ke-i
    bool2 = seg == Data.Segmen(i);                      % bandingin Segmen ke-1 dan ke-i
    
    % Kalo ketemu yg file sama & segmen sama
    if (bool)&&(bool2)
        % Ground truth
        if (strcmp('Arteri',char(Data.Vessel(i))))
            GT = [GT; 1];
        elseif (strcmp('Vena',char(Data.Vessel(i))))
            GT = [GT; -1];
        else
            GT = [GT; 0];
        end
        % Ambil nilai diameter & angle to OD
        arrD = [arrD; Data.Diameter(i)]; 
        arrT = [arrT; Data.AngleToOD(i)];
        arrX = [arrX; Data.Baris(i)];
        arrY = [arrY; Data.Kolom(i)];
    else
        % voting kalo mostly vena then segmennya jadi vena & so on
        % utk GT
        if (sum(GT,'all')<-1)
            GT = -1;
        elseif (sum(GT,'all')>1)
            GT = 1;
        else
            GT = 0;
        end
        % kalo jumlah hitungan diamternya lebih dari numCL, dihitung.
        if length(arrD)>numCL
            % ------------------------------------------------------------- modif D 
%             arrD = arrD(n:end-n,:);
            % ------------------------------------------------------------- end of modif D 
            D = mean(arrD(:)); 
            theta = mean(arrT(:));
            meanX = mean(arrX(:));
            meanY = mean(arrY(:));
            FN = [FN; name];
            T = [T; seg, D, std(arrD), min(arrD), median(arrD), mode(arrD), max(arrD), theta, meanX, meanY,GT];
        end
        % disimpan data2 yg perlu
        GT = []; Classifier = []; arrD = []; arrT = []; arrX=[]; arrY=[];
        name = Data.Filename(i); seg = Data.Segmen(i); 
        arrD = [arrD; Data.Diameter(i)]; 
        arrT = [arrT; Data.AngleToOD(i)];
        arrX = [arrX; Data.Baris(i)];
        arrY = [arrY; Data.Kolom(i)];
    end

    if (i==row)
        if (sum(GT,'all')<-1)
            GT = -1;
        elseif (sum(GT,'all')>1)
            GT = 1;
        else
            GT = 0;
        end

        if length(arrD)>numCL
            % ------------------------------------------------------------- modif D 
%             arrD = arrD(n:end-n,:);
            % ------------------------------------------------------------- end of modif D 
            D = mean(arrD(:)); 
            theta = mean(arrT(:));
            meanX = mean(arrX(:));
            meanY = mean(arrY(:));
            FN = [FN; name];
            T = [T; seg, D, std(arrD), min(arrD), median(arrD), mode(arrD), max(arrD), theta, meanX, meanY,GT];
        end
    end
end
Tab = array2table(T,...
        'VariableNames',{'Segmen','mean_D','std_D','min_D','med_D','mod_D','max_D','theta','meanX','meanY','GT'});
Tab.Filename = FN;
Tab = [Tab(:,end),Tab(:,1:end-1)];
disp('Done Tab');
%% PEMISAH ANTAR FILE (BEDA FILE)
[row col] = size(Tab);
beda_file = []; j = 1; first = 1;
name = Tab.Filename(1);
for i=1:row
    bool = strcmp(char(name),char(Tab.Filename(i)));
    if (bool~=1)
        last = i-1;
        beda_file = [beda_file; j, first, last];
        j = j+1; first = i; 
        name = Tab.Filename(i);
    end
    if (i==row)
        last = i;
        beda_file = [beda_file; j, first, last];
    end
end 
disp('Done pisah beda file');
%% Pemilihan Segmen yg harus digabung
Data = [];
for h=10:10 % AVRDB nih
    part = Tab(beda_file(h,2):beda_file(h,3),:);
    idx = find(part.GT(:)>0); Art = part(idx,1:end);
    idx = find(part.GT(:)<0); Vein = part(idx,1:end);
    Art = sortrows(Art, 3, 'descend'); 
    Vein = sortrows(Vein, 3, 'descend');
end
%% READ FILE
clear all;
% Membaca/input tabel dari csv (pilih file) 
% Feature_A/V
[fnameCSV, pnameCSV] = uigetfile('*.csv','Please select Data file(s)', 'MultiSelect', 'off');
fullnameCSV = strcat(pnameCSV,fnameCSV);
Data = readtable(fullnameCSV);
%% MULAI Visualisasi A/V
% Baca citra VESSEL SEGMENTATION
[fnameSEG, pnameSEG] = uigetfile('*.tiff','Please select VESSEL SEGMENTATION file(s)', 'MultiSelect', 'on'); %tiff
fullnameSEG = strcat(pnameSEG,fnameSEG);
%%
% Baca file ROI
[fnameROI, pnameROI] = uigetfile({'*.mat'},'Please select ROI file(s)', 'MultiSelect', 'on');
fullnameROI = strcat(pnameROI,fnameROI);
%%
% Raw AVRDB
[fnameRAW, pnameRAW] = uigetfile('*.JPG','Please select RAW IMAGE(s)', 'MultiSelect', 'on');
fullnameRAW = strcat(pnameRAW,fnameRAW);
n_RAW = length(fullnameRAW);
%% Khusus AVRDB
[fnameVES, pnameVES] = uigetfile({'*.jpg'},'Please select VESSEL IMAGE(s)', 'MultiSelect', 'on');
fullnameVES = strcat(pnameVES,fnameVES);
%% VISUALISASI SEGMEN ARTERI & VENA SELURUHNYA -- Cara Thea
for h=1:1
    Morph = imread(cell2mat(fullnameSEG(h)));
%     vessImage = imread(cell2mat(fullnameVES(h)));
%     vessImage = rgb2gray(vessImage);
%     T_vI = graythresh(vessImage);
%     vI = imbinarize(vessImage, T_vI);
%     Morph = 1-vI;
%     Morph = imread(cell2mat(fullnameRAW(h)));
    [row col warna] = size(Morph);
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
    Morph = ROI.*Morph;

%     if h<10
%         name = strcat('image0', num2str(h))
%     else
        name = strcat('image', num2str(h))
%     end

    name = 'IM000001';

    blank = logical(ones(row,col));

    figure, 
    imshow(Morph),
%     imshow(blank),
    hold on,
    idx = (find((strcmp(Data.Filename(:),name))&(strcmp(Data.Vessel(:),'Vena')))); 
    scatter(Data.Kolom(idx),Data.Baris(idx),1,'.','blue');
    idx = (find((strcmp(Data.Filename(:),name))&(strcmp(Data.Vessel(:),'Arteri')))); 
    scatter(Data.Kolom(idx),Data.Baris(idx),1,'.','red'); 
    
    hold off
    figure, 
    imshow(Morph), hold on

    position = [Data.Kolom(idx), Data.Baris(idx)];
%     position =  [1 50; 100 50];
    value = [Data.Segmen(idx)];
    RGB = insertText(Morph,position,value);
   
    figure, 
    imshow(RGB)
%     figure, 
%     imshow(Morph),
% %     imshow(blank),
%     hold on,
%     Segmen = Data.Segmen(idx);
%     n_seg = max(Segmen(:));
%     txt = 1:n_seg;
%     text(Data.Kolom(idx), Data.Segmen(idx));
%     hold off

    % save figures
%     temp_name = cell2mat(fnameSEG(h));
%     temp_name = temp_name(1,5:length(temp_name)-5);
%     fout = strcat('AV_',temp_name,'_Telea_AfterPenyambunganV4.jpg');
%     saveas(gcf,fout);
end
