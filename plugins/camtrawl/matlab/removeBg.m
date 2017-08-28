function img2 = removeBg(img)

F = fft2(double(img));
F = fftshift(F); % Center FFT

FF = abs(F); % Get the magnitude
FF = log(FF+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
FF = mat2gray(FF); % Use mat2gray to scale the image between 0 and 1
% figure, imshow(FF,[]); % Display the result

mask = zeros(size(FF));
[M,N] = size(FF);
win_size = [round(M/20),round(N/20)];
thresh = 0.1;

peakBW = imregionalmax(FF);

scale = 0.3;
mask(round(M/2)-round(scale*win_size(1)):round(M/2)+round(scale*win_size(1)),round(N/2)-round(scale*win_size(2)):round(N/2)+round(scale*win_size(2))) = peakBW(round(M/2)-round(scale*win_size(1)):round(M/2)+round(scale*win_size(1)),round(N/2)-round(scale*win_size(2)):round(N/2)+round(scale*win_size(2)));

scale2 = 8;
mask(1:scale2*win_size(1),:) = 1;
mask(end-scale2*win_size(1)+1:end,:) = 1;
mask(:,1:scale2*win_size(2)) = 1;
mask(:,end-scale2*win_size(2)+1:end) = 1;
mask = mask>0;
% figure, imshow(mask,'border','tight')


F(mask) = 0;
F2 = ifftshift(F);
img2 = real(ifft2(F2))/255;

% p = 0.5;
% img2 = p*img2+double(img)/255*(1-p);