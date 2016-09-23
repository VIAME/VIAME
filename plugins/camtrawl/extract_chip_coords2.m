function [object_set,object_chips] = extract_chip_coords2(targets,im)

[m,n]=size(imL);
h=size(targets,1);
object_set=zeros(h,1:5);

for i=1:h
    xs=targets(i,1:4);
    ys=targets(i,5:8);
    xm=mean(xs);
    dx=(max(xs)-min(xs))/2;
    ym=mean(ys);
    dy=(max(xs)-min(xs))/2;
    xminbox=max(floor(xm-dx*1.5),1);
    xmaxbox=min(floor(xm+dx*1.5),m);
    yminbox=max(floor(ym-dy*1.5),1);
    ymaxbox=min(floor(ym+dy*1.5),n);
    object_set(i,1:5)=[xminbox,yminbox,xmaxbox,ymaxbox, 1];
    
    xtot=xmaxbox-xminbox+1;
    ytot=ymaxbox-yminbox+1;
    object_chips(i).chip=uint8(im(xminbox:xmaxbox,yminbox:ymaxbox));
    
end