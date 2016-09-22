function [mask, imd, gmm] = gmm_background_remove(gmm,im,factor)

if size(im,3)==3 %this is a color image, convert to grayscale
    im=rbg2gray(im);
end
% resample using mean
imd=imresize(im,1/factor);
fg_d = step(gmm, uint8(imd));
ele = strel('disk',2);
fg_d = imopen(fg_d,ele);
fg_d = imdilate(fg_d,ele);
mask=fg_d>0;