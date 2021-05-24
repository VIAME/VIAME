function [xp,dxpdom,dxpdT,dxpdf,dxpdc,dxpdk,dxpdalpha] = project_points2(X,om,T,f,c,k,alpha)

%project_points2.m
%
%[xp,dxpdom,dxpdT,dxpdf,dxpdc,dxpdk] = project_points2(X,om,T,f,c,k,alpha)
%
%Projects a 3D structure onto the image plane.
%
%INPUT: X: 3D structure in the world coordinate frame (3xN matrix for N points)
%       (om,T): Rigid motion parameters between world coordinate frame and camera reference frame
%               om: rotation vector (3x1 vector); T: translation vector (3x1 vector)
%       f: camera focal length in units of horizontal and vertical pixel units (2x1 vector)
%       c: principal point location in pixel units (2x1 vector)
%       k: Distortion coefficients (radial and tangential) (4x1 vector)
%       alpha: Skew coefficient between x and y pixel (alpha = 0 <=> square pixels)
%
%OUTPUT: xp: Projected pixel coordinates (2xN matrix for N points)
%        dxpdom: Derivative of xp with respect to om ((2N)x3 matrix)
%        dxpdT: Derivative of xp with respect to T ((2N)x3 matrix)
%        dxpdf: Derivative of xp with respect to f ((2N)x2 matrix if f is 2x1, or (2N)x1 matrix is f is a scalar)
%        dxpdc: Derivative of xp with respect to c ((2N)x2 matrix)
%        dxpdk: Derivative of xp with respect to k ((2N)x4 matrix)
%
%Definitions:
%Let P be a point in 3D of coordinates X in the world reference frame (stored in the matrix X)
%The coordinate vector of P in the camera reference frame is: Xc = R*X + T
%where R is the rotation matrix corresponding to the rotation vector om: R = rodrigues(om);
%call x, y and z the 3 coordinates of Xc: x = Xc(1); y = Xc(2); z = Xc(3);
%The pinehole projection coordinates of P is [a;b] where a=x/z and b=y/z.
%call r^2 = a^2 + b^2.
%The distorted point coordinates are: xd = [xx;yy] where:
%
%xx = a * (1 + kc(1)*r^2 + kc(2)*r^4 + kc(5)*r^6)      +      2*kc(3)*a*b + kc(4)*(r^2 + 2*a^2);
%yy = b * (1 + kc(1)*r^2 + kc(2)*r^4 + kc(5)*r^6)      +      kc(3)*(r^2 + 2*b^2) + 2*kc(4)*a*b;
%
%The left terms correspond to radial distortion (6th degree), the right terms correspond to tangential distortion
%
%Finally, convertion into pixel coordinates: The final pixel coordinates vector xp=[xxp;yyp] where:
%
%xxp = f(1)*(xx + alpha*yy) + c(1)
%yyp = f(2)*yy + c(2)
%
%
%NOTE: About 90 percent of the code takes care of computing the Jacobian matrices
%
%
%Important function called within that program:
%
%rodrigues.m: Computes the rotation matrix corresponding to a rotation vector
%
%rigid_motion.m: Computes the rigid motion transformation of a given structure


if nargin < 7,
   alpha = 0;
   if nargin < 6,
      k = zeros(5,1);
      if nargin < 5,
         c = zeros(2,1);
         if nargin < 4,
            f = ones(2,1);
            if nargin < 3,
               T = zeros(3,1);
               if nargin < 2,
                  om = zeros(3,1);
                  if nargin < 1,
                     error('Need at least a 3D structure to project (in project_points.m)');
                     return;
                  end;
               end;
            end;
         end;
      end;
   end;
end;


[m,n] = size(X);

[Y,dYdom,dYdT] = rigid_motion(X,om,T);


inv_Z = 1./Y(3,:);

x = (Y(1:2,:) .* (ones(2,1) * inv_Z));


bb = (-x(1,:) .* inv_Z)'*ones(1,3);
cc = (-x(2,:) .* inv_Z)'*ones(1,3);


dxdom = zeros(2*n,3);
dxdom(1:2:end,:) = ((inv_Z')*ones(1,3)) .* dYdom(1:3:end,:) + bb .* dYdom(3:3:end,:);
dxdom(2:2:end,:) = ((inv_Z')*ones(1,3)) .* dYdom(2:3:end,:) + cc .* dYdom(3:3:end,:);

dxdT = zeros(2*n,3);
dxdT(1:2:end,:) = ((inv_Z')*ones(1,3)) .* dYdT(1:3:end,:) + bb .* dYdT(3:3:end,:);
dxdT(2:2:end,:) = ((inv_Z')*ones(1,3)) .* dYdT(2:3:end,:) + cc .* dYdT(3:3:end,:);


% Add distortion:

r2 = x(1,:).^2 + x(2,:).^2;

dr2dom = 2*((x(1,:)')*ones(1,3)) .* dxdom(1:2:end,:) + 2*((x(2,:)')*ones(1,3)) .* dxdom(2:2:end,:);
dr2dT = 2*((x(1,:)')*ones(1,3)) .* dxdT(1:2:end,:) + 2*((x(2,:)')*ones(1,3)) .* dxdT(2:2:end,:);


r4 = r2.^2;

dr4dom = 2*((r2')*ones(1,3)) .* dr2dom;
dr4dT = 2*((r2')*ones(1,3)) .* dr2dT;


r6 = r2.^3;

dr6dom = 3*((r2'.^2)*ones(1,3)) .* dr2dom;
dr6dT = 3*((r2'.^2)*ones(1,3)) .* dr2dT;


% Radial distortion:

cdist = 1 + k(1) * r2 + k(2) * r4 + k(5) * r6;

dcdistdom = k(1) * dr2dom + k(2) * dr4dom + k(5) * dr6dom;
dcdistdT = k(1) * dr2dT + k(2) * dr4dT + k(5) * dr6dT;
dcdistdk = [ r2' r4' zeros(n,2) r6'];


xd1 = x .* (ones(2,1)*cdist);

dxd1dom = zeros(2*n,3);
dxd1dom(1:2:end,:) = (x(1,:)'*ones(1,3)) .* dcdistdom;
dxd1dom(2:2:end,:) = (x(2,:)'*ones(1,3)) .* dcdistdom;
coeff = (reshape([cdist;cdist],2*n,1)*ones(1,3));
dxd1dom = dxd1dom + coeff.* dxdom;

dxd1dT = zeros(2*n,3);
dxd1dT(1:2:end,:) = (x(1,:)'*ones(1,3)) .* dcdistdT;
dxd1dT(2:2:end,:) = (x(2,:)'*ones(1,3)) .* dcdistdT;
dxd1dT = dxd1dT + coeff.* dxdT;

dxd1dk = zeros(2*n,5);
dxd1dk(1:2:end,:) = (x(1,:)'*ones(1,5)) .* dcdistdk;
dxd1dk(2:2:end,:) = (x(2,:)'*ones(1,5)) .* dcdistdk;



% tangential distortion:

a1 = 2.*x(1,:).*x(2,:);
a2 = r2 + 2*x(1,:).^2;
a3 = r2 + 2*x(2,:).^2;

delta_x = [k(3)*a1 + k(4)*a2 ;
   k(3) * a3 + k(4)*a1];


%ddelta_xdx = zeros(2*n,2*n);
aa = (2*k(3)*x(2,:)+6*k(4)*x(1,:))'*ones(1,3);
bb = (2*k(3)*x(1,:)+2*k(4)*x(2,:))'*ones(1,3);
cc = (6*k(3)*x(2,:)+2*k(4)*x(1,:))'*ones(1,3);

ddelta_xdom = zeros(2*n,3);
ddelta_xdom(1:2:end,:) = aa .* dxdom(1:2:end,:) + bb .* dxdom(2:2:end,:);
ddelta_xdom(2:2:end,:) = bb .* dxdom(1:2:end,:) + cc .* dxdom(2:2:end,:);

ddelta_xdT = zeros(2*n,3);
ddelta_xdT(1:2:end,:) = aa .* dxdT(1:2:end,:) + bb .* dxdT(2:2:end,:);
ddelta_xdT(2:2:end,:) = bb .* dxdT(1:2:end,:) + cc .* dxdT(2:2:end,:);

ddelta_xdk = zeros(2*n,5);
ddelta_xdk(1:2:end,3) = a1';
ddelta_xdk(1:2:end,4) = a2';
ddelta_xdk(2:2:end,3) = a3';
ddelta_xdk(2:2:end,4) = a1';



xd2 = xd1 + delta_x;

dxd2dom = dxd1dom + ddelta_xdom ;
dxd2dT = dxd1dT + ddelta_xdT;
dxd2dk = dxd1dk + ddelta_xdk ;


% Add Skew:
xd3 = [xd2(1,:) + alpha*xd2(2,:);xd2(2,:)];

% Compute: dxd3dom, dxd3dT, dxd3dk, dxd3dalpha

dxd3dom = zeros(2*n,3);
dxd3dom(1:2:2*n,:) = dxd2dom(1:2:2*n,:) + alpha*dxd2dom(2:2:2*n,:);
dxd3dom(2:2:2*n,:) = dxd2dom(2:2:2*n,:);
dxd3dT = zeros(2*n,3);
dxd3dT(1:2:2*n,:) = dxd2dT(1:2:2*n,:) + alpha*dxd2dT(2:2:2*n,:);
dxd3dT(2:2:2*n,:) = dxd2dT(2:2:2*n,:);
dxd3dk = zeros(2*n,5);
dxd3dk(1:2:2*n,:) = dxd2dk(1:2:2*n,:) + alpha*dxd2dk(2:2:2*n,:);
dxd3dk(2:2:2*n,:) = dxd2dk(2:2:2*n,:);
dxd3dalpha = zeros(2*n,1);
dxd3dalpha(1:2:2*n,:) = xd2(2,:)';



% Pixel coordinates:
if length(f)>1,
    xp = xd3 .* (f * ones(1,n))  +  c*ones(1,n);
    coeff = reshape(f*ones(1,n),2*n,1);
    dxpdom = (coeff*ones(1,3)) .* dxd3dom;
    dxpdT = (coeff*ones(1,3)) .* dxd3dT;
    dxpdk = (coeff*ones(1,5)) .* dxd3dk;
    dxpdalpha = (coeff) .* dxd3dalpha;
    dxpdf = zeros(2*n,2);
    dxpdf(1:2:end,1) = xd3(1,:)';
    dxpdf(2:2:end,2) = xd3(2,:)';
else
    xp = f * xd3 + c*ones(1,n);
    dxpdom = f  * dxd3dom;
    dxpdT = f * dxd3dT;
    dxpdk = f  * dxd3dk;
    dxpdalpha = f .* dxd3dalpha;
    dxpdf = xd3(:);
end;

dxpdc = zeros(2*n,2);
dxpdc(1:2:end,1) = ones(n,1);
dxpdc(2:2:end,2) = ones(n,1);


return;

% Test of the Jacobians:

n = 10;

X = 10*randn(3,n);
om = randn(3,1);
T = [10*randn(2,1);40];
f = 1000*rand(2,1);
c = 1000*randn(2,1);
k = 0.5*randn(5,1);
alpha = 0.01*randn(1,1);

[x,dxdom,dxdT,dxdf,dxdc,dxdk,dxdalpha] = project_points2(X,om,T,f,c,k,alpha);


% Test on om: OK

dom = 0.000000001 * norm(om)*randn(3,1);
om2 = om + dom;

[x2] = project_points2(X,om2,T,f,c,k,alpha);

x_pred = x + reshape(dxdom * dom,2,n);


norm(x2-x)/norm(x2 - x_pred)


% Test on T: OK!!

dT = 0.0001 * norm(T)*randn(3,1);
T2 = T + dT;

[x2] = project_points2(X,om,T2,f,c,k,alpha);

x_pred = x + reshape(dxdT * dT,2,n);


norm(x2-x)/norm(x2 - x_pred)



% Test on f: OK!!

df = 0.001 * norm(f)*randn(2,1);
f2 = f + df;

[x2] = project_points2(X,om,T,f2,c,k,alpha);

x_pred = x + reshape(dxdf * df,2,n);


norm(x2-x)/norm(x2 - x_pred)


% Test on c: OK!!

dc = 0.01 * norm(c)*randn(2,1);
c2 = c + dc;

[x2] = project_points2(X,om,T,f,c2,k,alpha);

x_pred = x + reshape(dxdc * dc,2,n);

norm(x2-x)/norm(x2 - x_pred)

% Test on k: OK!!

dk = 0.001 * norm(k)*randn(5,1);
k2 = k + dk;

[x2] = project_points2(X,om,T,f,c,k2,alpha);

x_pred = x + reshape(dxdk * dk,2,n);

norm(x2-x)/norm(x2 - x_pred)


% Test on alpha: OK!!

dalpha = 0.001 * norm(k)*randn(1,1);
alpha2 = alpha + dalpha;

[x2] = project_points2(X,om,T,f,c,k,alpha2);

x_pred = x + reshape(dxdalpha * dalpha,2,n);

norm(x2-x)/norm(x2 - x_pred)
