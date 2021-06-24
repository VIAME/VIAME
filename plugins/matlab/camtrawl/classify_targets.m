function [spc_code,probs]=classify_targets(targets,im)

for i=1:size(targets)
    xm=mean(targets(i,1:4));
    ym=mean(targets(i,5:8));
    xminbox=max(floor(xm-dx*1.5),1);
    xmaxbox=min(floor(xm+dx*1.5),m);
    yminbox=max(floor(ym-dy*1.5),1);
    ymaxbox=min(floor(ym+dy*1.5),n);
    
    chip=im(yminbox:ymaxbox,xminbox:xmaxbox);
    
    chip_modify = removeBg(chip);
    
    [pred,prob,rotateRect_r] = transfer_image_predict2(uint8(chip),uint8(255*chip_modify));
    
    spc_code(i)=pred;
    probs(i)=prob;
end
