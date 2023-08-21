function tepian = pengambilantepi(Gambar)
width = 45;
height = 45;
kmax = pi/2;
f = sqrt( 2 );
delta =  pi/3 ;
% Gambar = imread('21_training.tif');
img_green = Gambar(:,:,2);
img_in_gray = im2double(rgb2gray(Gambar));
img_in_h = histeq(img_in_gray);
% imshow(img_in_h);
img_out = zeros(size(img_in_gray,1), size(img_in_gray,2), 8);
for u = 0 : 9
     GW = GaborWaveletTepian ( width, height, kmax, f, u, 6, delta ); % Create the Gabor wavelets
%       figure;
%        subplot( 1, 8, u + 1 ),imshow ( real(GW),[]);
      img_out(:,:,u+1) = imfilter(img_in_gray, GW, 'symmetric');
end
% default superposition method, L2-norm
img_out_disp = sum(abs(img_out).^2, 3).^0.5;
img_out_disp = img_out_disp./max(img_out_disp(:));
% normalize
% figure;
% imshow(img_out_disp),title('deteksi tepi');

imgthreshold = im2bw(img_out_disp,0.2); %avrdb 0.4
% figure,imshow(imgthreshold),title('threshold');
imgtepi = bwareaopen(imgthreshold, 10); % avrdb 30
% figure,imshow(imgtepi),title('open');
se = strel('disk',5);

tepian = imdilate(imgtepi,se);
% figure,imshow(tepian),title('tepihasil');
end
