%   Processing_with_species_id.m
%   Automated processing of fish in left/right image pairs.
%   Output is a set of lengths from all the frames and hauls specified.
%   The lengths are computed by: 
%               a. Performing a global (full image) threshold subtraction
%                  based on a running averaged background to create a
%                  binary image with potential fish objects.  This is done
%                  for the right and left frame, individually;
%               b. Each object is labeled, fit with an oriented
%                  rectangle;
%               c. The coordinates of the boxes are used to select local
%                  object frames and threshold according to each individual
%                  fish's grayscale pixel distribution.  The threshold is
%                  set to be the mean + 1 standard deviation of the
%                  gaussian fit to the grayscale values.  Then relabeling
%                  and re-fitting with an oriented rectange.
%               d. Fish are matched between the left and right image using
%                  error threshold from transforming points into 3
%                  dimensions and then back to 2D pixel space.
%               e. The length between the mid points of the fit, matched
%                  boxes are used to find the length of the fish.
%
%
%   Functions required to run:
%                                extract_targets2.m
%                                find_match.m
%                                calssify_species.m

%
%   3D transformation functions (from camera clibration toolbox - Bouguet 2008):
%                                stereo_triangulation.m
%                                project_points2.m 
%                                   rodrigues.m
%                                   rigid_motion.m
%                                   normalize_pixel.m
%                                       comp_distortion_oulu.m
%                                       comp_distortion.m

clear all; clc; close all;

%% DEFINE PARAMETERS

% Set path for image data
dep='image_data\';
% Camera specific folder names
left_name='left';
right_name='right';
% Set path for matlab mat file save
save_path='';

% load calibration data
%load calibration_parameters
load cal_201608

% Select start and end frame numbers for images that have fish passing through in both left and right.
% Enter into ranges the haul number, start frame number, end frame number
ranges=[2400,2600];

% Lengthing will determine and save lengths of fish processed
saving=true;

% Select whether to perform species ID
species_id=false;

% Plotting will allow user to view the images that are being processed
plotting=true;

% Adjust the threshold value for left (thL) and right (thR) so code will
% select most fish without including non-fish objects (e.g. the net)
thL=20;
thR=20;

% Species classes
% This is fixed for each training set, so it will remain the same throughout an entire survey
% pollock, salmon unident., rockfish unident.
sp_numbs=[21740,23202,30040];

% Threshold for errors between before & after projected
% points to make matches between left and right
max_err=[50,50];

% Minimum aspect ratio for filtering out non-fish objects
min_aspect=3.5;

% Maximum aspect ratio for filtering out non-fish objects
max_aspect=7.5;

% minimum number of pixels to keep a section, after sections 
% are found by component function
min_size=2000;        

% limits accepable targets to be within this region 
%[left, bottom, width, height] - depends on image size!
ROI=[12,12,412*2-24,309*2-24]; 

% number to increment between frames
by_n=1; 

% Factor to reduce the size of the image for processing
factor=2;


GMM_start_idx = 1;  % the start frame of GMM
GMM_detector_L = vision.ForegroundDetector(...
       'NumTrainingFrames', 30, ...
       'InitialVariance', 30*30);  
GMM_detector_R = vision.ForegroundDetector(...
       'NumTrainingFrames', 30, ...
       'InitialVariance', 30*30);  

%% Set up figure
if plotting
    f1=figure;
    set(f1,'Position',[ 22         502        1243         420]);
end


%% BEGIN PROCESSING





    haulstr='072';
    depPath=[dep,'\images\'];
    % find information on all the frames requested using dir
    filesL=dir([depPath,left_name,'\*.jpg']);
    filesR=dir([depPath,right_name,'\*.jpg']);
    imt=imread([depPath,left_name,'\',filesL(1).name]);
    frames=ranges(1):ranges(2);
    first=true;
    % initialize result vector
    lengths=[];
    % initialize background
%     imL=imread([depPath,left_name,'\',filesL(1).name]);
%     imR=imread([depPath,right_name,'\',filesR(1).name]);
%     if size(imL,3)==3; %this is a color image, convert to grayscale
%         imL=uint8(mean(double(imL(:,:,2:3)),3));
%     end
%     if size(imR,3)==3; %this is a color image, convert to grayscale
%         imR=uint8(mean(double(imR(:,:,2:3)),3));
%     end
%     
%     bgL=resize_image(imL,factor);
%     bgR=resize_image(imR,factor);

    %% Loop through frames in the current haul
    tic
    for o = 2:length(frames)
        if plotting==0
            try
                h = waitbar(o/length(frames));
            catch
            end
        end
        % work on left frame first
        % initialize targets for this frame
        targetsL=[];
        targetsR=[];
        % come up with fae number in string padded with 0s
        frstr=num2str(frames(o));%
        if frames(o)<10000
            frstr=[repmat('0',1,5-length(num2str(frames(o)))),num2str(frames(o))];
        end
        %% %%%%%   left image initialization    %%%%%
        file=dir([depPath,left_name,'\',frstr,'*.jpg']);
        
        if isempty(file)
            continue
        else
            imL=imread([depPath,left_name,'\',file.name]);
        end
        % average channels in the case of color images
        if size(imL,3)==3; %this is a color image, convert to grayscale
            imL=uint8(mean(double(imL(:,:,2:3)),3));
        end
        % resample using mean 
        imLd=imresize(imL,1/factor);
        % imLd=resize_image(imL,factor);
        % subtract background
        fg_dL = step(GMM_detector_L, uint8(imLd));
        ele = strel('disk',2);
        fg_dL = imopen(fg_dL,ele);
        fg_dL = imdilate(fg_dL,ele);
        % extract regions
        targetsL=extract_targets2(fg_dL>0,imLd,min_size,ROI,min_aspect,max_aspect,factor);
        if size(targetsL,1)>0;% valid targets found in left image
            %% %%%%%   right image initialization    %%%%%
            file=dir([depPath,right_name,'\',frstr,'*.jpg']);
            if isempty(file)
                continue
            else
                imR=imread([depPath,right_name,'\',file.name]);
            end
            % average channels in the case of color images
            if size(imR,3)==3; 
                imR=uint8(mean(double(imR(:,:,2:3)),3));
            end
            % resample using mean 
            imRd=imresize(imR,1/factor);
            % subtract background
            fg_dR = step(GMM_detector_R, uint8(imRd));
            ele = strel('disk',2);
            fg_dR = imopen(fg_dR,ele);
            fg_dR = imdilate(fg_dR,ele);
            %[fg_dR,bgR]=background_subtract(imRd,bgR,thR);
            % label regions
            targetsR=extract_targets2(fg_dR>0,imRd, min_size,ROI,min_aspect,max_aspect,factor);
            if size(targetsR,1)>0;% valid targets found in right image
                % find matching stereo targets, return final target vectors
                [targetsL,targetsR,er,L,Z,DZ]=find_match(targetsL,targetsR,max_err,Cal);
                if species_id % do this on left image only
                    [spc_code,prob]=classify_targets(targetsL,imL);
                else
                    spc_code=zeros(size(targetsL,1),1);
                    prob=zeros(size(targetsL,1),1);
                end
                % write results to result array
                % loop though targets
                for i = 1:size(targetsL)
                    % to display:
                    current_frame=frames(o);
                    fishLength=L(i);
                    % to save:
                    fishRange=Z(i);
                    cL=targetsL(i,1:8);
                    cR=targetsR(i,1:8);
                    Err=er(i);
                    Lar=targetsL(i,9);
                    Rar=targetsR(i,9);
                    LboxL=targetsL(i,10);
                    LboxR=targetsR(i,10);
                    WboxL=targetsL(i,11);
                    WboxR=targetsR(i,11);
                    aveL=mean([targetsL(i,10),targetsR(i,10)]);
                    aveW=mean([targetsL(i,11),targetsR(i,11)]);
                    dz=DZ(i);
                    lengths=[lengths;[current_frame,fishLength,fishRange,Err,cL,cR,Lar,Rar,LboxL,LboxR,WboxL,WboxR,aveL,aveW,dz,spc_code(i),prob(i)]];
                end% result writing loop
                 if plotting
                    if size(targetsL,1)>0
                        [m,n]=size(imR);
                        hL=subplot(1,2,1);
                        imagesc(imL); colormap gray
                        title(['LEFT  ',frstr])
                        for i = 1:size(targetsL,1)
                            line(targetsL(i,[1:4,1]),targetsL(i,[5:8,5]),'Color',[1,0,0],'linewidth',2);
                            text(mean(targetsL(i,[1:4]))+25,mean(targetsL(i,[5:8])),[num2str(i)],'Color',[1,0,0])
                        end
                        hR=subplot(1,2,2);
                        imagesc(imR); colormap gray
                        title(['RIGHT  ',frstr])
                        for i = 1:size(targetsR,1)
                            line(targetsR(i,[1:4,1]),targetsR(i,[5:8,5]),'Color',[1,0,0],'linewidth',2)
                            text(mean(targetsR(i,[1:4]))+25,mean(targetsR(i,[5:8])),[num2str(i)],'Color',[1,0,0])
                        end
                        pause(0.5)
                    end
                end
            end% valid targets in right
        end% valid targets in left
        disp(['Completed frame ',frstr,' - measured ',num2str(size(targetsL)),' fish'])
    end% frame loop
    toc
    %save?
    close all;
    if saving
        % get run parameters
        params=[ranges,max_err,min_size,ROI,min_aspect,max_aspect,factor, by_n];
        eval(['save ',pwd,save_path,'\Haul_',haulstr,' lengths params Cal']);
    end
