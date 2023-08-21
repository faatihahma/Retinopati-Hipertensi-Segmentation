    % Path

% mat path
labels_path = ('D:\Bismillah TA\Kode\Share code\ImageTraining\Sushi_Ekstraksi Fitur\AVRDB_thresholdVmax\cbgabung');

% File Names
labels_names = getMultipleImagesFileNames(labels_path);

% File Labels
segment_labels = cell(length(labels_names), 1);

%%
allData  = struct();
% tabel = zeros(length(i),18);
for i= 1 : length(segment_labels)
    data = load(fullfile(labels_path, labels_names{i} ));
    allData = fieldnames(data);
     for iField = 1:numel(allData)
         aField = allData{iField};
         if isfield(allData, aField)             % Attach new data:
             allData.(aField) = [allData.(aField), Data.(aField)];
         else
             allData.(aField) = data.(aField);
         end
     end
end
% data = cell2mat(data); % combine all small column vectors into one big one
% fout = strcat('gabung_1', '.mat');
% save(fout, 'allData')
save('AllData.mat', '-struct', 'allData');
%% 
FileList = dir(fullfile('D:\Bismillah TA\Kode\Share code\ImageTraining\Sushi_Ekstraksi Fitur\AVRDB_thresholdVmax\cbgabung', '*.mat'));  % List of all MAT files
allData  = struct();
for iFile = 1:numel(FileList)               % Loop over found files
  Data   = load(fullfile('D:\Bismillah TA\Kode\Share code\ImageTraining\Sushi_Ekstraksi Fitur\AVRDB_thresholdVmax\cbgabung', FileList(iFile).name));
  Fields = fieldnames(Data);
  for iField = 1:numel(Fields)              % Loop over fields of current file
    aField = Fields{iField};
    allData.(aField) = [Data.(aField)];
    if isfield(allData, aField)             % Attach new data:
        allData.(aField) = Data.(aField);
    end
      
       
%        % [EDITED]
%        % The orientation depends on the sizes of the fields. There is no
%        % general method here, so maybe it is needed to concatenate 
%        % vertically:
%        % allData.(aField) = [allData.(aField); Data.(aField)];
%        % Or in general with suiting value for [dim]:
%        % allData.(aField) = cat(dim, allData.(aField), Data.(aField));
    
       
  end
end
save('cb3citra.mat', '-struct', 'allData');
%%
x = struct()
for i= 1 : length(segment_labels)
    y = load(fullfile(labels_path, labels_names{i} ));
    vrs = fieldnames(y);
    for k = 1:length(vrs)
        field = vrs{k}
        x.(field) = [y.field];
    end
end
save('Data.mat','-struct','x')
%%
a = load('EktraksiFitur_IM001154.mat');
b = load('EktraksiFitur_IM001192.mat');
c = load('EktraksiFitur_IM001218.mat');
d = load('EktraksiFitur_IM001686.mat');
e = load('EktraksiFitur_IM001749.mat');
% f = load('EktraksiFitur_IM001069.mat');
% g = load('EktraksiFitur_IM001097.mat');
% h = load('EktraksiFitur_IM000715.mat');
% i = load('EktraksiFitur_IM001034.mat');
% j = load('EktraksiFitur_IM001070.mat');
% l = load('EktraksiFitur_IM001127.mat');
% m = load('EktraksiFitur_IM001193.mat');
% n = load('EktraksiFitur_IM001219.mat');
% o = load('EktraksiFitur_IM001750.mat');
% p = load('EktraksiFitur_IM001792.mat');

vrs = fieldnames(a);
if ~isequal(vrs,fieldnames(a))
    error('Different variables in these MAT-files')
end
for k = 1:length(vrs)
%     field = vrs{k}
    x.(vrs{k}) = [a.(vrs{k});b.(vrs{k});c.(vrs{k});d.(vrs{k});e.(vrs{k})];% f.(vrs{k});g.(vrs{k})];%h.(vrs{k});i.(vrs{k});j.(vrs{k})]; 
                 %l.(vrs{k});m.(vrs{k});n.(vrs{k});o.(vrs{k});p.(vrs{k})];
end
% Save result in a new file
save('Data_5_exud.mat','-struct','x')