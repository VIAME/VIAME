function d = mahalanobis(X,varargin)
%Inputs:
% X: samples from a multivariate distribution
% F: frequencies of samples
% Output:
% d: Mahalanobis distance of samples

% [N D] = size(X);
D = size(X,2);
if D>1
    if isempty(varargin)
        [X,~,IC] = unique(X,'rows','legacy');
        lastloc = find(diff([0;sort(IC)]));
        F = diff([lastloc;lastloc(end)+1]);
    else
        F = varargin{1}(:);
    end
    N = size(X,1);
    F = F/sum(F);
    d = zeros(N,1);
    if any(diff(F)~=0)
        M = F'*X;
    else
        M = sum(X,1);
    end
    % M = sum(X.*repmat(F,1,D),1);
    X = X-repmat(M,N,1);
    S = X'*(X.*repmat(F,1,D));
    if (sum(sum(abs(diff(S))))==0)
        S = diag(diag(S));
    end
%     Q = inv(S);
    for i = 1:N
         d(i) = (X(i,:)/S)*X(i,:)';
    end
    d = sqrt(abs(d));
else
    if isempty(varargin)
        [X,~,IC] = unique(X,'rows','legacy');
        lastloc = find(diff([0;sort(IC)]));
        F = diff([lastloc;lastloc(end)+1]);
    else
        F = varargin{1}(:);
    end
    F = F/sum(F);
    X = X - F'*X;
    S = X'*(X.*F);
    Q = 1/S;
    d = sqrt(X.*(Q*X));
end
if isempty(varargin)
d = d(IC);
end
