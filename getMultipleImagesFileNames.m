% Open multiple files from a given directory
function allNames = getMultipleImagesFileNames(directory)
    % Get all file names
    allFiles = dir(directory);
    % Get only the names of the images inside the folder
    allNames = {allFiles.name};
%     allNames(strcmp(allNames, '...')) = [];
    allNames(strcmp(allNames, '..')) = [];
    allNames(strcmp(allNames, '.')) = [];
    allNames(strcmp(allNames, 'Thumbs.db')) = [];
%     allNames = removeFileNamesWithExtension(allNames, 'mat'); %untuk
%     deteksi file .mat perlu di comment terlebih dahulu, kalau image bisa
%     di uncomment
end
