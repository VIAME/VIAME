function [rect labels] = imOrientedBox(img, theta)
%IMORIENTEDBOX Minimum-area oriented bounding box of particles in image
%
%   OBB = imOrientedBox(IMG);
%   Computes the minimum area oriented bounding box of labels in image IMG.
%   IMG is either a binary or a label image. 
%   The result OBB is a N-by-11 array, containing the four rectangle
%   corners as 4 x positions and 4 y positions (1-8), larger of the two
%   aspect ratios for the length and width (9), rectangle length (10), and
%   rectangle width (11) for 1-N particles in the image.  
%
%   The original code made an N-5 array, containing
%   the center, the length, the width, and the orientation of the bounding
%   box of each particle in the image.  The orientation is given in radians, in the direction of the greatest
%   box axis.
%
%   OBB = imOrientedBox(IMG, NDIRS);
%   OBB = imOrientedBox(IMG, DIRSET);
%   Specifies either the number of directions to use for computing boxes
%   (default is 180), or the set of directions.
%
%   Example
%   imOrientedBox
%
%   See also
%   imFeretDiameter
%
% ------
% Author: David Legland
% e-mail: david.legland@grignon.inra.fr
% Created: 2011-02-07,    using Matlab 7.9.0.529 (R2009b)
% Copyright 2011 INRA - Cepia Software Platform.

% choose default values for number of directions
if ~exist('theta', 'var')
    theta = 180;
end

% if theta is scalar, create an array of directions (in degrees)
if isscalar(theta)
    theta = linspace(0, 180, theta+1);
    theta = theta(1:end-1);
end
nTheta = length(theta);

% extract the set of labels, and remove label for background
labels = unique(img(:));
labels(labels==0) = [];

nLabels = length(labels);

% allocate memory for result
rect = zeros(nLabels, 11);

for i=1:nLabels
    % extract points of the current particle
    [y x] = find(img==labels(i));
    
    % keep only points of the convex hull
    try
        inds = convhull(x, y);
        x = x(inds);
        y = y(inds);
    catch ME %#ok<NASGU>
        % an exception can occur if points are colinear.
        % in this case we transform all points
    end

    % compute convex hull centroid, that corresponds to approximate
    % location of rectangle center
    xc = mean(x);
    yc = mean(y);
    
    % recenter points (should be better for numerical accuracy)
    x = x - xc;
    y = y - yc;

    % allocate memory for result of Feret Diameter computation
    fd = zeros(1, nTheta);

    % iterate over orientations
    for t = 1:nTheta
        % convert angle to radians, and change sign (to make transformed
        % points aligned along x-axis)
        theta2 = -theta(t) * pi / 180;

        % compute only transformed x-coordinate
        x2  = x * cos(theta2) - y * sin(theta2);

        % compute diameter for extreme coordinates
        xmin    = min(x2);
        xmax    = max(x2);

        % store result (add 1 pixel to consider pixel width)
        fd(t)   = xmax - xmin + 1;
    end

    % compute area of enclosing rectangles with various orientations
    feretArea = fd(:, 1:end/2).*fd(:, end/2+1:end);
    
    % find the orientation that produces to minimum area rectangle
    [minArea indMinArea] = min(feretArea, [], 2); %#ok<ASGLU>

    % convert index to angle (in degrees)
    indMin90 = indMinArea + nTheta/2;
    if fd(indMinArea) < fd(indMin90)
        thetaMax = theta(indMin90);
    else
        thetaMax = theta(indMinArea);
    end

    % elongation in direction of rectangle length
    thetaRect = thetaMax * pi / 180;
    x2  = x * cos( thetaRect) - y * sin(-thetaRect);
    y2  = x * sin(-thetaRect) + y * cos( thetaRect);

    % compute extension along main directions
    xmin = min(x2);    xmax = max(x2);
    ymin = min(y2);    ymax = max(y2);
    x=[xmin,xmin,xmax,xmax];
    y=[ymin,ymax,ymax,ymin];
    
    %[t,r]=cart2pol(x,y);
    t = atan2(y,x);
    r = sqrt(x.^2+y.^2);
    x = r.*cos(t+thetaRect);
    y = r.*sin(t+thetaRect);
    
    x=x+xc;
    y=y+yc;
    
    rectLength  = xmax - xmin + 1;
    rectWidth   = ymax - ymin + 1;
    ar=[rectLength/rectWidth,rectWidth/rectLength];
    % concatenate rectangle data
    rect(i,:) = [x,y,max(ar),rectLength,rectWidth];
end