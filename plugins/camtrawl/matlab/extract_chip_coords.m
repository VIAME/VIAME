function object_set = extract_chip_coords(targets)

[m,n]=size(imL);

for i=1:size(targets)
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
    
    
end