function [OBB]=extract_targets2(mask,img,min_size,ROI,min_aspect,max_aspect,factor)
% this function finds contiguous regions, filters the targets by size,
% applies an oriented boundingbox, and filters by aspect ratio
OBB=[];
%4 way connected components algorithm
[labeled_mask,label]=bwlabel(mask);
if label>0 % there's some targets
    % remove small targets
    labeled_mask_filt1=zeros(size(labeled_mask));
    keepers=0;
    fish_size=[];
    fish_location=[];
    fish_extent=[];
    for i=1:label
        [x,y]=find(labeled_mask==i);
        if length(x)>=min_size% filter size
            if (min(x)>=ROI(2) && max(x)<=(ROI(4)-ROI(2)) && min(y)>=ROI(1) && max(y)<=(ROI(3)-ROI(1))) % filter edges
                keepers=keepers+1;
                labeled_mask_filt1(find(labeled_mask==i))=keepers;
                fish_size=[fish_size,length(x)];
                fish_location=[fish_location,[mean(x);mean(y)]];
                fish_extent=[fish_extent,[max(x)-min(x);max(y)-min(y)]];
            end
        end
    end
    
    % fit the bounding boxes
    OBB = imOrientedBoxx(labeled_mask_filt1);
    
    % filter by aspect ratio
    if size(OBB,1)>0
        for i = 1:size(OBB,1)
            aspectRatio(i)=max([OBB(i,10)./OBB(i,11),OBB(i,11)./OBB(i,10)]);
        end
        OBB=OBB(find(aspectRatio>min_aspect & aspectRatio<max_aspect),:);
        
        % scale up coordiantes back to original image size
        OBB(:,1:8)=OBB(:,1:8)*factor;
        % also store x and y min and max for the target 'chip'
        
        
    end
    

end