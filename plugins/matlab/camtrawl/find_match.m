function [targetsL,targetsR,er,L,R,DZ] = find_match(coordsL,coordsR,max_err,Cal)

% Function to find fish that match between the left and right frames given
% an input of coordsL, coordsR, a max_err and the calibration data for
% coordinate transformations, Cal.  This is a temp one for the version of
% autolengths that just processes lengths by using boxes.  More is output
% of this version than the others to include the length and parameters
% range, R, error from the matching and change in distance from the camera across the length of the
% fish, DZ.

numL=size(coordsL,1);
numR=size(coordsR,1);

row_count=0;
lme=length(max_err);

for i=1:numL
    for j=1:numR
        
        %   The result OBB is a N-by-11 array, containing the four rectangle
        %   corners as 4 x positions and 4 y positions (1-8), larger of the two
        %   aspect ratios for the length and width (9), rectangle length (10), and
        %   rectangle width (11) for 1-N particles in the image.  
        %
        % [ x1, x2, x3, x4, y1, y2, y3, y4, ar, rect-len, rect-width] 
        % [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ]
        x1=[mean(coordsL(i,1:2)),mean(coordsL(i,3:4))];
        y1=[mean(coordsL(i,5:6)),mean(coordsL(i,7:8))];
        
        x2=[mean(coordsL(i,2:3)),mean(coordsL(i,[1,4]))];
        y2=[mean(coordsL(i,6:7)),mean(coordsL(i,[8,5]))];
        
        if sqrt((x1(1)-x1(2))^2+(y1(1)-y1(2))^2)>sqrt((x2(1)-x2(2))^2+(y2(1)-y2(2))^2)
            xL=[x1;y1];
        else
            xL=[x2;y2];
        end
        clear x1 x2 y1 y2
        
        x1=[mean(coordsR(j,1:2)),mean(coordsR(j,3:4))];
        y1=[mean(coordsR(j,5:6)),mean(coordsR(j,7:8))];
        
        x2=[mean(coordsR(j,2:3)),mean(coordsR(j,[1,4]))];
        y2=[mean(coordsR(j,6:7)),mean(coordsR(j,[8,5]))];
        
        
        if sqrt((x1(1)-x1(2))^2+(y1(1)-y1(2))^2)>sqrt((x2(1)-x2(2))^2+(y2(1)-y2(2))^2)
            xR=[x1;y1];
        else
            xR=[x2;y2];
        end
        [crp,xLi]=max(xL(1,:));
        [crp,xRi]=max(xR(1,:));
        if xLi~=xRi
            xR=fliplr(xR);
        end
        clear x1 x2 y1 y2
        

        % Triangulation output:
        %   XL: 3xN matrix of coordinates of the points in the left camera reference frame
        %   XR: 3xN matrix of coordinates of the points in the right camera reference frame
        % Note: XR and XL are related to each other through the rigid motion equation: XR = R * XL + T, where R = rodrigues(om)
        %
        [XL,XR] = stereo_triangulation(xL,xR,...
            Cal.om,Cal.T,Cal.fc_left,Cal.cc_left,...
            Cal.kc_left,Cal.alpha_c_left,Cal.fc_right,...
            Cal.cc_right,Cal.kc_right,Cal.alpha_c_right);
        
        xlp = project_points2(XL,[0,0,0]',[0 0 0]',Cal.fc_left,Cal.cc_left,Cal.kc_left,Cal.alpha_c_left);
        xrp = project_points2(XR,[0,0,0]',[0 0 0]',Cal.fc_right,Cal.cc_right,Cal.kc_right,Cal.alpha_c_right);

        % Average reprojection error.
        %  xL and xR both contain two points, hence xL makes 2 errors and xR
        %  makes 2 errors.  So there are a total of 4 errors -> divide by 4.
        E(i,j)=(sum(sqrt((xL(1,:)-xlp(1,:)).^2+(xL(2,:)-xlp(2,:)).^2)+...
                    sqrt((xR(1,:)-xrp(1,:)).^2+(xR(2,:)-xrp(2,:)).^2)))/4;

        
        % Compute the length on the diagonal (using only the left coords) of
        % the of the 3D bounding box This approximates the fish length. (Not
        % sure why we are dividing by 10 here, units?)
        len=sqrt((XL(1,1)-XL(1,2))^2+(XL(2,1)-XL(2,2))^2+(XL(3,1)-XL(3,2))^2)/10;
        z=mean([XL(3,1),XL(3,2)]);
        dz=abs(XL(3,2)-XL(3,1));
        
        if lme==1
            if E(i,j)<max_err && E(i,j)>0 && XL(3,1)>0 && XL(3,2)>0 && XR(3,1)>0 && XR(3,2)>0
                row_count=row_count+1;
                matched(row_count,1:2)=[i,j];
                er(row_count)=E(i,j);
                L(row_count)=len;
                R(row_count)=z;
                DZ(row_count)=dz;
            end
        end
        
        if lme==2
            if len<=15
                if E(i,j)<max_err(1) && E(i,j)>0 && XL(3,1)>0 && XL(3,2)>0 && XR(3,1)>0 && XR(3,2)>0
                    row_count=row_count+1;
                    matched(row_count,1:2)=[i,j];
                    er(row_count)=E(i,j);
                    L(row_count)=len;
                    R(row_count)=z;
                    DZ(row_count)=dz;
                end
            else
                if E(i,j)<max_err(2) && E(i,j)>0 && XL(3,1)>0 && XL(3,2)>0 && XR(3,1)>0 && XR(3,2)>0
                    row_count=row_count+1;
                    matched(row_count,1:2)=[i,j];
                    er(row_count)=E(i,j);
                    L(row_count)=len;
                    R(row_count)=z;
                    DZ(row_count)=dz;
                end
            end
        end
                
        clear xlp xrp xL xR XL XR
    end
end
if row_count==0
    matched=[];
    er=[];
    L=[];
    R=[];
    DZ=[];
else
    % remove multiple object matches and keep
    % the match with lowest error
    iL=find(diff(matched(:,1))==0);
    lL=length(iL);
    subt=0;
    if lL>0
        for i=1:lL
            [k,~]=max(er(iL(i)-subt:iL(i)+1-subt));
            ind1=find(er==k);
            matched(ind1,:)=[];
            er(ind1)=[];
            L(ind1)=[];
            R(ind1)=[];
            DZ(ind1)=[];
            subt=subt+1;
        end
    end
    iR=find(diff(matched(:,2))==0);
    lR=length(iR);
    subt=0;
    if lR>0
        for i=1:lR
            [k,~]=max(er(iR(i)-subt:iR(i)+1-subt));
            ind1=find(er==k);
            matched(ind1,:)=[];
            er(ind1)=[];
            L(ind1)=[];
            R(ind1)=[];
            DZ(ind1)=[];
            subt=subt+1;
        end
    end
end
% line up the target arrays so that the match and are same size.
tempcoordsL=[];tempcoordsR=[];
for i = 1:size(matched,1)
    tempcoordsL(i,:)=coordsL(matched(i,1),:);
    tempcoordsR(i,:)=coordsR(matched(i,2),:);
end
targetsL=tempcoordsL;
targetsR=tempcoordsR;


