%% Hitung AVR Knudtson

% READ FILE
clear all;
% Membaca/input tabel dari csv (pilih file) --> afterCorr
[fnameCSV, pnameCSV] = uigetfile('*.csv','Please select Data file(s)', 'MultiSelect', 'off');
fullnameCSV = strcat(pnameCSV,fnameCSV);
Data = readtable(fullnameCSV);

% Baca file Vesel Hasil Classifier COLAB
[fnameCSV, pnameCSV] = uigetfile('*.csv','Please select Vessel file(s)', 'MultiSelect', 'off');
fullnameCSV = strcat(pnameCSV,fnameCSV);
Vessel = readtable(fullnameCSV);

%% Gabungin Segmen
% Segmen yg sama dihitung mean diameter, voting arteri/vena.
% n=20;
% [row col] = size(Data); numCL = 60; % INSPIRE
[row col] = size(Data); numCL = 2; % AVRDB
T = []; GT = []; Classifier = []; FN = []; arrD =   []; arrVes = []; arrT = [];
name = Data.Filename(1); seg = Data.Segmen(1);
for i=1:row
    bool = strcmp(char(name),char(Data.Filename(i)));   % bandingin Filename ke-1 dan ke-i
    bool2 = seg == Data.Segmen(i);                      % bandingin Sgemen ke-1 dan ke-i
    
    % Kalo ketemu yg file sama & segmen sama
    % ganti arteri/vena/NB --> 1/-1/0
    if (bool)&&(bool2)
        % Ground truth
        if (strcmp('Arteri',char(Data.Vessel(i))))
            GT = [GT; 1];
        elseif (strcmp('Vena',char(Data.Vessel(i))))
            GT = [GT; -1];
        else
            GT = [GT; 0];
        end
        % Hasil Classifier Colab
        if (strcmp('Arteri',char(Vessel.Vessel(i))))
            Classifier = [Classifier; 1];
        elseif (strcmp('Vena',char(Vessel.Vessel(i))))
            Classifier = [Classifier; -1];
        else
            Classifier = [Classifier; 0];
        end
        % Ambil nilai diameter & angle to OD
        arrD = [arrD; Data.Diameter(i)]; arrT = [arrT; Data.AngleToOD(i)];
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
        % utk classifier
        if (sum(Classifier,'all')<-1)
            Classifier = -1;
        elseif (sum(Classifier,'all')>1)
            Classifier = 1;
        else
            Classifier = 0;
        end
        % kalo jumlah hitungan diamternya lebih dari numCL, dihitung.
        if length(arrD)>numCL
            % ------------------------------------------------------------- modif D 
%             arrD = arrD(n:end-n,:);
            % ------------------------------------------------------------- end of modif D
            D = mean(arrD(:)); theta = mean(arrT(:));
            FN = [FN; name];
            T = [T; seg, D, std(arrD), min(arrD), median(arrD), mode(arrD), max(arrD), theta, GT, Classifier];
        end
        % disimpan data2 yg perlu
        GT = []; Classifier = []; arrD = []; arrT = [];
        name = Data.Filename(i); seg = Data.Segmen(i); arrD = [arrD; Data.Diameter(i)]; arrT = [arrT; Data.AngleToOD(i)];
    end
    if (i==row)
        if (sum(GT,'all')<-1)
            GT = -1;
        elseif (sum(GT,'all')>1)
            GT = 1;
        else
            GT = 0;
        end
        
        if (sum(Classifier,'all')<-1)
            Classifier = -1;
        elseif (sum(Classifier,'all')>1)
            Classifier = 1;
        else
            Classifier = 0;
        end
        
        if length(arrD)>numCL
            % ------------------------------------------------------------- modif D 
%             arrD = arrD(n:end-n,:);
            % ------------------------------------------------------------- end of modif D
            D = mean(arrD(:)); theta = mean(arrT(:));
            FN = [FN; name];
            T = [T; seg, D, std(arrD), min(arrD), median(arrD), mode(arrD), max(arrD), theta, GT, Classifier];
        end
    end
end  
Tab = array2table(T,...
        'VariableNames',{'Segmen','mean_D','std_D','min_D','med_D','mod_D','max_D','theta','GT','Classifier'});
Tab.Filename = FN;
Tab = [Tab(:,end),Tab(:,1:end-1)];

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

%% HITUNG AVR KNUDTSON -- INSPIRE & AVRDB
dataAVR = []; Tab.Ket = Tab.GT==Tab.Classifier; 
dataArt = []; dataVein = [];
for h=1:99 % AVRDB nih
    part = Tab(beda_file(h,2):beda_file(h,3),:);
    idx = find(part.Classifier(:)>0); Art = part(idx,1:end);
    idx = find(part.Classifier(:)<0); Vein = part(idx,1:end);
    Art = sortrows(Art, 3, 'descend'); 
    Vein = sortrows(Vein, 3, 'descend');
    
    i=1;
    while i<size(Art,1)-1
        j=i+1;
        while j<size(Art,1)
            difD = abs(Art.mean_D(i)-Art.mean_D(i+1)); difT = abs(Art.theta(i)-Art.theta(i+1));
            if difD<=1.5 && difT<=0.2
                Art(j,:) = [];
            else
                j=j+1;
            end
        end
        i=i+1;
    end
    i=1;
    while i<size(Vein,1)-1
        j=i+1;
        while j<size(Vein,1)
            difD = abs(Vein.mean_D(i)-Vein.mean_D(i+1)); difT = abs(Vein.theta(i)-Vein.theta(i+1));
            if difD<=1.5 && difT<=0.2
                Vein(j,:) = [];
            else
                j=j+1;
            end
        end
        i=i+1;
    end
    
%     idx = find(Art.mean_D<11); Art(idx,:) = [];
%     idx = find(Vein.mean_D<13); Vein(idx,:) = [];

%     m=1;
%     while (Art.mean_D(m)<Vein.mean_D(m)) && m<size(Art,1) && m<size(Vein,1)
%         m = m+1;
%     end
    
    if (height(Art)>=6) && (height(Vein)>=6)
        m=6;
    else
        m=1;
        while (Art.mean_D(m)<Vein.mean_D(m)) && m<size(Art,1) && m<size(Vein,1)
            m = m+1;
        end
    end

    % AVR Knudtson
    [N, CRAE, CRVE, AVR] = calAVR_Knudtson(Art.mean_D(1:m),Vein.mean_D(1:m));
    dataArt = [dataArt; Art(1:N,:)]; 
    dataVein = [dataVein; Vein(1:N,:)]; 
    dataAVR = [dataAVR; h, N, CRAE, CRVE, AVR];
end

TabAVR = array2table(dataAVR,'VariableNames',{'Citra','N','CRAE','CRVE','AVR'});
TabAVR.N = [];

namaFile = input('Nama file, contoh: namafile.csv == ','s')
writetable(TabAVR,namaFile);

%% SAMPAI SINI AJA UTK HITUNG AVR -----------------------------------------
% Selanjutnya utk visualisasi/plot









%% MULAI Visualisasi A/V
% Baca citra VESSEL SEGMENTATION
[fnameSEG, pnameSEG] = uigetfile('*.tiff','Please select VESSEL SEGMENTATION file(s)', 'MultiSelect', 'on');
fullnameSEG = strcat(pnameSEG,fnameSEG);

% Baca file ROI
[fnameROI, pnameROI] = uigetfile({'*.mat'},'Please select ROI file(s)', 'MultiSelect', 'on');
fullnameROI = strcat(pnameROI,fnameROI);

% Raw AVRDB
[fnameRAW, pnameRAW] = uigetfile('*.JPG','Please select RAW IMAGE(s)', 'MultiSelect', 'on');
fullnameRAW = strcat(pnameRAW,fnameRAW);
n_RAW = length(fullnameRAW);

%% Visualisasi Cara Yuni
Art = []; Vein = [];
for h=1:length(beda_file)
    name = Tab.Filename(beda_file(h,2));
    idx = find(strcmp(dataArt.Filename(:),name)); seg = dataArt.Segmen(idx); N = length(seg);
    idx = find(strcmp(Data.Filename(:),name)); part = Data(idx,1:4);
    for i=1:N
        idx = find(part.Segmen==seg(i));
        Art = [Art; (h)*ones(length(idx),1), part.Baris(idx), part.Kolom(idx)];
    end
    
    idx = find(strcmp(dataVein.Filename(:),name)); seg = dataVein.Segmen(idx); N = length(seg);
    idx = find(strcmp(Data.Filename(:),name)); part = Data(idx,1:4);
    for i=1:N
        idx = find(part.Segmen==seg(i));
        Vein = [Vein; (h)*ones(length(idx),1), part.Baris(idx), part.Kolom(idx)];
    end
    
end

%% Visualisasi Segmen A/V utk hitung AVR -- Cara Yuni
for h=2:50
    Morph = imread(cell2mat(fullnameSEG(h)));
    [row col] = size(Morph);
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
    % blank = zeros(2048,2392); 
    figure, 
    imshow(Morph.*ROI), 
    hold on,
    idx = find(Art(:,1)==h); 
    scatter(Art(idx,3),Art(idx,2),1,'.','red'),
    idx = find(Vein(:,1)==h); 
    scatter(Vein(idx,3),Vein(idx,2),1,'.','blue'), hold off

    % save figures
    temp_name = cell2mat(fnameSEG(h));
    temp_name = temp_name(1,5:length(temp_name)-5);
    fout = strcat(temp_name,'_CRAECRVE_FMMonly_m6.jpg');
    saveas(gcf,fout);
end

%% VISUALISASI SEGMEN ARTERI & VENA SELURUHNYA -- Cara Thea
for h=1
%     Morph = imread(cell2mat(fullnameSEG(h)));
    Morph = imread(cell2mat(fullnameRAW(h)));
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
    Morph = Morph.*ROI;

%     if h<10
%         name = strcat('image0', num2str(h))
%     else
        name = strcat('image', num2str(h))
%     end

    name = 'IM005321';

    blank = logical(ones(row,col));

    figure, 
%     imshow(Morph),
    imshow(blank),
    hold on,
    idx = (find((strcmp(Data.Filename(:),name))&(strcmp(Data.Vessel(:),'Vena')))); 
    scatter(Data.Kolom(idx),Data.Baris(idx),1,'.','blue');
    idx = (find((strcmp(Data.Filename(:),name))&(strcmp(Data.Vessel(:),'Arteri')))); 
    scatter(Data.Kolom(idx),Data.Baris(idx),1,'.','red'), hold off

    % save figures
%     temp_name = cell2mat(fnameSEG(h));
%     temp_name = temp_name(1,5:length(temp_name)-5);
%     fout = strcat('AV_',temp_name,'_Telea_AfterPenyambunganV4.jpg');
%     saveas(gcf,fout);
end

%% PLOT AV -- Cara Thea
Data.Classifier = Vessel.Vessel;
Data.Ket = strcmp(Data.Vessel,Data.Classifier);
h=1
    Raw = imread(cell2mat(fullnameRAW(h)));
    [row col apa] = size(Raw);
    blank = logical(ones(row,col));
    
    name = 'IM000716';
    figure, 
%     imshow(Morph),
    imshow(blank),
    hold on,
    idx = (find((strcmp(Data.Filename(:),name))&(strcmp(Data.Vessel(:),'Vena'))&(Data.Ket==1))); 
    scatter(Data.Kolom(idx),Data.Baris(idx),1,'.','blue');
    idx = (find((strcmp(Data.Filename(:),name))&(strcmp(Data.Vessel(:),'Arteri'))&(Data.Ket==1))); 
    scatter(Data.Kolom(idx),Data.Baris(idx),1,'.','red');
    idx = (find((strcmp(Data.Filename(:),name))&(Data.Ket==0))); 
    scatter(Data.Kolom(idx),Data.Baris(idx),1,'.','green'),   hold off