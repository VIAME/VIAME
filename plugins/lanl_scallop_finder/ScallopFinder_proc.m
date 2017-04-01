function [detected_object_set,detected_object_classification] = ScallopFinder_proc(image)
[rows,cols,bands] = size(image);
image = image(:,:,[3 2 1]);
Rmax = 115;
Rmin = 10;
if bands>1
    X = rgb2hsv(double(reshape(image,rows*cols,bands))/255);
    red = 2*abs(X(:,1)-0.5);
    Tred = prctile(red,80);
%     red(red<Tred) = 0;
    red =red>=prctile(red,80);% boolean for redness
%     sat = X(:,2);
    sat = X(:,2)>mean(X(:,2))+1.5*std(X(:,2));% boolean for high saturation
else
    X = image(:);
    red = 1;
end

d = mahalanobis(X,ones(size(X,1),1));%pixel Mahalanobis distances
d = d.*red.*sat;
d = reshape(d,rows,cols);
d = medfilt2(d,[3 3]);%[9 9]

mind = min(d(:));
maxd = max(d(:));
image(:,:,4) = 255*(d-mind)/(maxd-mind);

d = im2bw(d,graythresh(d));
d = bwmorph(d,'close');
d = bwareaopen(d,100);


count = 0;
CR = zeros(2000,4);
disparity = zeros(2000,1);
% anomaly = zeros(2000,1);
image = double(image);
thetas = (0:0.1:2*pi)';
nt = numel(thetas);
for sigma = 1:3
    J = cannyedges(image,sigma);
    J = logical(J);
    J(:,[1:2 end-1:end]) = 0;
    J([1:2 end-1:end],:) = 0;
    C = contourchains(J);
    edgIdsLs = unique(C(:,3:4),'rows');% array of edgeids and their lengths
    T = prctile(edgIdsLs(:,2),90);% prctile cutoff on edge length
    C = C(C(:,4)>T,:);% retain edgepoints belonging to edges above cutoff
    C(:,3) = renum(C(:,3));% renumber retained edges
    ofst = [0;find(diff([C(:,3);0])~=0)];% offsets of edgepoints belonging to same edge

    for i = 1:numel(ofst)-1
        D = C(ofst(i)+1:ofst(i+1),1:2);
        n=ofst(i+1)-ofst(i);
        A = [2*D ones(n,1)];
        b = sum(D.^2,2);
        Par = (A\b)';
        Par(3) = sqrt(Par(1)^2+Par(2)^2+Par(3));
        radvar = sqrt(sum(bsxfun(@minus,D,Par(1:2)).^2,2))-Par(3);
        dev = max(abs(radvar))/Par(3);
        LD = C(ofst(i)+1,4);
        Peri = 2*pi*Par(3);
        support = LD/Peri;
        nrings = round(Par(3));
        if (dev<0.2)&&(support>0.2)&&(Par(3)>Rmin)&&(Par(3)<Rmax)
            count = count+1;
            insamp =zeros(nrings*nt,2);
            outsamp =zeros(nrings*nt,2);
            a = Par(1);
            b = Par(2);
            r = Par(3);
            for s = 1:ceil(nrings)
                insamp(nt*(s-1)+1:nt*s,1)=a + (r-s)*sin(thetas);
                insamp(nt*(s-1)+1:nt*s,2)=b + (r-s)*cos(thetas);
                outsamp(nt*(s-1)+1:nt*s,1)=a + (r+s)*sin(thetas);
                outsamp(nt*(s-1)+1:nt*s,2)=b + (r+s)*cos(thetas);
            end
            insamp = round(insamp);
            outsamp = round(outsamp);
            outbds =(insamp(:,1)<1)|(insamp(:,2)<1)|(insamp(:,1)>rows)|(insamp(:,2)>cols);
            insamp(outbds,:) = [];
            outbds =(outsamp(:,1)<1)|(outsamp(:,2)<1)|(outsamp(:,1)>rows)|(outsamp(:,2)>cols);
            outsamp(outbds,:) = [];
            insamp = (insamp(:,2)-1)*rows+insamp(:,1);
            outsamp = (outsamp(:,2)-1)*rows+outsamp(:,1);
%             id = d(unique(insamp));
%             od = d(unique(outsamp));
%             anomaly(count) = mean(id);
%             disparity(count) = anomaly(count)-mean(od);
            insamp = unique(insamp);
            outsamp = unique(outsamp);
            id = sum(d(insamp));
            od = sum(d(outsamp));
            disparity(count) = max(0,id-od);

            CR(count,1:3) = Par;
            CR(count,4) = id/(pi*Par(3)*Par(3));
        end

    end

end
CR = CR(1:count,:);
disparity = disparity(1:count);
% anomaly = anomaly(1:count);
% Tdisparity = max(min(disparity(disparity>0)),prctile(disparity,50));
% if isempty(Tdisparity)
%     Tdisparity = Inf;
% end
% Tanomaly = min(2,prctile(anomaly,50));
% inds = (anomaly>Tanomaly)&(disparity>Tdisparity);
inds = (disparity>0);
CR = CR(inds,:);
Nold = Inf;
Nnew = size(CR,1);
count = 0;
if Nnew>0
    while Nold>Nnew
        count = count+1;
        Nold = Nnew;
        DT = delaunayTriangulation(CR(:,1:2));
        E = edges(DT);
        dist = sqrt(sum((CR(E(:,1),1:2)-CR(E(:,2),1:2)).^2,2));
        R1 = CR(E(:,1),3);
        R2 = CR(E(:,2),3);
        V = true(size(CR,1),1);
        for i = 1:size(E,1);
            if dist(i)<=(R1(i)+R2(i))
                [~,ind] = min([R1(i) R2(i)]);
                    V(E(i,ind)) = false;
            end
        end
        CR = CR(V,:);
        Nnew = sum(V);
    end
end
L = size(CR,1);

detected_object_set = [CR(:,2) - CR(:,3) ...
                       CR(:,1) - CR(:,3) ...
                       CR(:,2) + CR(:,3) ...
                       CR(:,1) + CR(:,3) CR(:,3)];% box and confidence for detection
detected_object_classification = struct();
for i = 1:L
    detected_object_classification(i,1).name='Scallop';
    detected_object_classification(i,1).score= CR(i,4);
end

%% %%%%%%%%%%%%%%%%%%%%%SUBFUNCTIONS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edges = cannyedges(I,sigma)
%   cannyedges takes an intensity or a binary image I (of class double) as its input, and returns a
%   binary image 'edges' of edge pixels of the same size as I, with 1's where the function
%   finds edges in I and 0's elsewhere.  sigma is the standard deviation of the Gaussian filter.
m = size(I,1);
n = size(I,2);
if size(size(I),2) == 3
    bands = size(I,3);
else
    bands = 1;
end

edges = false(m,n);% initialize output edge map:

% parameters:
GaussianDieOff = 0.0001;
EdgePixelRatio = 0.3; % Used for selecting thresholds
ThresholdRatio = 0.0;  % Low thresh is this fraction of the high.

% gaussian and derivative of gaussian for convolution:
pw = 1:30; % possible widths
ssq = sigma*sigma;
width = find(exp(-(pw.*pw)/(2*sigma*sigma))>GaussianDieOff,1,'last');
if isempty(width)
    width = 1;  % the user entered a really small sigma
end

t = (-width:width);
gau = exp(-(t.*t)/(2*ssq))/(2*pi*ssq);     % the gaussian 1D filter

% Find the directional derivative of 2D Gaussian (along X-axis)
% Since the result is symmetric along X, we can get the derivative along
% Y-axis simply by transposing the result for X direction.
[x,y]=meshgrid(-width:width,-width:width);
dgau2D=-x.*exp(-(x.*x+y.*y)/(2*ssq))/(pi*ssq);

mag = zeros(m,n);
for i = 1:bands
    %smooth the image out
    aSmooth=imfilter(I(:,:,i),gau,'conv','replicate');         % run the filter across rows
    aSmooth=imfilter(aSmooth,gau','conv','replicate');  % and then across columns

    %apply directional derivatives
    ax = imfilter(aSmooth, dgau2D, 'conv','replicate');
    ay = imfilter(aSmooth, dgau2D', 'conv','replicate');
    mag = max(mag, sqrt((ax.*ax) + (ay.*ay)));
end

magmax = max(mag(:));
if magmax>0
    mag = mag / magmax;   % normalize
end
bins = 128;
% Select hysteresis thresholds
counts=imhist(mag, bins);
highThresh = find(cumsum(counts) > (1-EdgePixelRatio)*m*n,1) /bins;
lowThresh = ThresholdRatio*highThresh;

% Non-maximum supression:
% idxStrong = [];
idxStrong = cell(m*n,1);
count = 0;
for dir = 1:4
    idxLocalMax = cannyFindLocalMaxima(dir,ax,ay,mag);
    idxWeak = idxLocalMax(mag(idxLocalMax) > lowThresh);
    edges(idxWeak)=1;
    count = count+1;
    idxStrong{count} = idxWeak(mag(idxWeak) > highThresh);
%     idxStrong = [idxStrong; idxWeak(mag(idxWeak) > highThresh)];
end
idxStrong = idxStrong(1:count);
idxStrong = cat(1,idxStrong{:});

rstrong = rem(idxStrong-1, m)+1;
cstrong = floor((idxStrong-1)/m)+1;
edges = bwselect(edges, cstrong, rstrong, 8);
edges = bwmorph(edges, 'thin', Inf);  % Thin edge image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function idxLocalMax = cannyFindLocalMaxima(direction,ix,iy,mag)

[m,n] = size(mag);

switch direction
    case 1
        idx = find((iy<=0 & ix>-iy)  | (iy>=0 & ix<-iy));
    case 2
        idx = find((ix>0 & -iy>=ix)  | (ix<0 & -iy<=ix));
    case 3
        idx = find((ix<=0 & ix>iy) | (ix>=0 & ix<iy));
    case 4
        idx = find((iy<0 & ix<=iy) | (iy>0 & ix>=iy));
end
% Exclude exterior pixels
if ~isempty(idx)
    v = mod(idx,m);
    extIdx = v==1 | v==0 | idx<=m | (idx>(n-1)*m);
    idx(extIdx) = [];
end

ixv = ix(idx);
iyv = iy(idx);
gradmag = mag(idx);
% Linear interpolations for interior pixels
switch direction
    case 1
        d = abs(iyv./ixv);
        gradmag1 = mag(idx+m).*(1-d) + mag(idx+m-1).*d;
        gradmag2 = mag(idx-m).*(1-d) + mag(idx-m+1).*d;
    case 2
        d = abs(ixv./iyv);
        gradmag1 = mag(idx-1).*(1-d) + mag(idx+m-1).*d;
        gradmag2 = mag(idx+1).*(1-d) + mag(idx-m+1).*d;
    case 3
        d = abs(ixv./iyv);
        gradmag1 = mag(idx-1).*(1-d) + mag(idx-m-1).*d;
        gradmag2 = mag(idx+1).*(1-d) + mag(idx+m+1).*d;
    case 4
        d = abs(iyv./ixv);
        gradmag1 = mag(idx-m).*(1-d) + mag(idx-m-1).*d;
        gradmag2 = mag(idx+m).*(1-d) + mag(idx+m+1).*d;
end
idxLocalMax = idx(gradmag>=gradmag1 & gradmag>=gradmag2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A,varargout] = renum(A, varargin)
% This function takes as input :
% 1)an array and remaps the entries from 1 to n,
% (where n is the number of unique entries in the array). It also determines an index
% column vector inds such that inds(v) restores the original
% value of the remapped entry v
% 2)a 2-d array of integers 1 to n and an index vector ind of length n and remaps the
% array entries such that i --> ind(i)
%This is useful for canonical indexing of the vertex set of a graph
if isempty(varargin)
S = size(A);
[B,~,J] = unique(A);
b = length(B);
V = (1:b)';
A = V(J);
A = reshape(A,S);
if nargout>0
    varargout{1} = B;
end
else
    A = varargin{1}(A);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
