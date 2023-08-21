function M = MaskAVRDB(V)

[row, col] = size(V);
A = im2bw(V,0.2); 
A(1,:) = zeros(1,col); A(row,:) = zeros(1,col);
A(:,1) = zeros(row,1); A(:,col) = zeros(row,1);
A = imfill(A,'holes');
imshow(A)

C = A;
for i=1:25
    C = masked(C);
end
% imshowpair(A,A.*C,'montage')

M = C;
end




