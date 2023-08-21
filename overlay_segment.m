
% Path

% % Ground Truth (GT) path
labels_path = ('D:\Bismillah TA\Kode\Share code\ImageTraining\Vess');

% Segmentation Result Path
segment_path = fullfile('D:\Bismillah TA\Kode\Segmentasi Hasil');

% Original RGB Image Path / Sgementasi yuni / Exud cek      
original_path = fullfile('D:\Bismillah TA\Kode\Share code\ImageTraining\Image');

% File Names
original_names = getMultipleImagesFileNames(original_path);
segment_names = getMultipleImagesFileNames(segment_path);
labels_names = getMultipleImagesFileNames(labels_path);

% File Labels
segment_labels = cell(length(segment_names), 1);
gt_labels = cell(length(labels_names), 1);
original_labels = cell(length(original_names), 1);

%% Overlaying with iteration
DataPixel = [];
for i = 1 : length(segment_names)
    
    % Read RGB Image
    O = imread(fullfile(original_path, original_names{i} ));

    % Read Segmentation Result Image
    I = imread(fullfile(segment_path, segment_names{i} ));
%     K = imread(fullfile(labels_path, labels_names{i} ));


    % Change Image Type to Logical
%     BW = imbinarize(I);
    
    % If image I is already logical change to this :
    BW = I;

    % Read ground truth Image
    vI = imread(fullfile(labels_path, labels_names{i}));
    vessImage = rgb2gray(vI);
    T_vI = graythresh(vessImage);
    vI = imbinarize(vessImage, T_vI);
    vI = 1-vI;
%     Resize GT image to the same size as segmentation result
%     J = imresize(J,size(I));
% % 
%     % Change to logical Image if J is not logical type
%     BWJ = imbinarize(J);

    % if image J already logical change to this
    BWJ = vI;
    
    % Confusion Matrix
%     tp = sum(BW(:) == 1 & BWJ(:) == 1);
%     fn = sum(BW(:) == 0 & BWJ(:) == 1);
%     tn = sum(BW(:) == 0 & BWJ(:) == 0);
%     fp = sum(BW(:) == 1 & BWJ(:) == 0);
    
%     idx1 = find(I==1); 
%     TP = length(find(vI(idx1)==1)); FP = length(find(vI(idx1)==0));
%     idx2 = find(I==0);
%     TN = length(find(vI(idx2)==0)); FN = length(find(vI(idx2)==1));
    
    % Overlay GT and Segmentations
    overlay = imfuse(BWJ,BW,'falsecolor','ColorChannels',[1 2 0]);
%     figure, imshow(overlay);
    temp_name = cell2mat(segment_names(i));
    temp_name = temp_name(1,1:length(temp_name)-4);
    fout = strcat('Overlay_',temp_name, '.jpg');
    imwrite(overlay, fout, 'jpg');
    % Yellow= TP
    % Red = FN
    % Green = FP
%     idx1 = find(O==1);
%     EX = length(find(I(idx1)==1)); TX = length(find(I(idx1)==0));
%     idx2 = find(O==0);
%     NV = length(find(I(idx2)==0)); V = length(find(I(idx2)==1));
%     DataPixel = [DataPixel; i, EX, TX, NV, V]
    % Resize original RGB Image
%     O = imresize(O,size(I));

    % Overlay RGB Image with overlay result
    overlayrgb = imfuse(O,overlay,'blend');
%     figure, imshow(overlayrgb);
%     temp_name = cell2mat(segment_names(i));
%     temp_name = temp_name(1,1:length(temp_name)-4);
%     fout = strcat('RGB_', temp_name, '.jpg');
%     imwrite(overlayrgb, fout, 'jpg'); 
end
