function [object_set, chips] = extract_chip(targets, im)

for i=1:size(targets)
    xm=mean(targets(i,1:4));
    ym=mean(targets(i,5:8));
    xminbox=max(floor(xm-dx*1.5),1);
    xmaxbox=min(floor(xm+dx*1.5),m);
    yminbox=max(floor(ym-dy*1.5),1);
    ymaxbox=min(floor(ym+dy*1.5),n);
    object_set(i,1:5)=[xminbox,yminbox,xmaxbox,ymaxbox, 1];
    
    chip=im(yminbox:ymaxbox,xminbox:xmaxbox);
    chips(i,:,:)=removeBg(chip);
    
end