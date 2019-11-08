function model = GMMz(X, K, options)
% Variational Bayesian inference for Gaussian mixture.
% Input: 
%   X: d x n data matrix
%   m: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or model structure
% Output:
%   label: 1 x n cluster label
%   model: trained model structure
%   L: variational lower bound
% Reference: Pattern Recognition and Machine Learning by Christopher M. Bishop (P.474)
% Written by Mo Chen (sth4nth@gmail.com).

fprintf('Variational Bayesian Gaussian mixture: running ... \n');
[n,d] = size(X);

O = ~isnan(X);


if nargin < 3
    
    options.tol = 1e-5;
    options.maxIter = 500;
    
end

if(sum(~O(:))>0)
    model = GMMz_missing(X, K, options);
    return;
end

model = kmeans_missing(X,1,5);
    
prior.alpha = 1;
prior.beta = 1;
prior.mu = model.mus(1,:);
prior.v = d+1;
prior.M = model.Sigmas(:,:,1);
    
prior.logW = -2*sum(log(diag(chol(prior.M))));




oldL = -inf;
model = init(X,K,prior);
fprintf('Optimizing ... \n');
for iter = 1:options.maxIter
    model = expect(X,model);
    model = maximize(X,model,prior);
    
    L = bound(X,model,prior)/n;
    fprintf('%d\t%f\n',iter,L);
    
    if abs((L - oldL)/oldL) < options.tol; break; end
    
    oldL = L;
end

model.weights = model.alpha/sum(model.alpha);

Nk = floor(model.alpha-1);

selected = Nk>2;

K = sum(selected);

weights = model.alpha(selected);
weights = weights/sum(weights);

mus = model.mus(selected,:);
Sigmas = model.Sigmas(:,:,selected);

model.K = K;
model.R = model.R(:,selected);
model.mus = mus;
model.Sigmas = Sigmas;
model.weights = weights;

function model = init(X, m, prior)
n = size(X,1);
if isstruct(m)  % init with a model
    model = m;
elseif numel(m) == 1  % random init k
    
    fprintf('Initializing using k-means with 20 iterations ... \n');
    [model, R] = kmeans_missing(X,m,20);
    model.R = R;
    
elseif all(size(m)==[1,n])  % init with labels
    label = m;
    k = max(label);
    model.R = full(sparse(1:n,label,1,n,k,n));
else
    error('ERROR: init is not valid.');
end
model = maximize(X,model,prior);

% Done
function model = maximize(X, model, prior)
alpha0 = prior.alpha;
beta0 = prior.beta;
mu0 = prior.mu;
v0 = prior.v;
M0 = prior.M;
R = model.R;

nk = sum(R); % 10.51
alpha = alpha0+nk; % 10.58
beta = beta0+nk; % 10.60
v = v0+nk; % 10.63
mus = bsxfun(@plus,R'*X,beta0*mu0);
mus = bsxfun(@rdivide,mus,beta'); % 10.61

[k,d] = size(mus);
U = zeros(d,d,k);
Sigmas = zeros(d,d,k);
logW = zeros(1,k);
r = sqrt(R);
for i = 1:k
    Xm = bsxfun(@minus,X,mus(i,:));
    Xm = bsxfun(@times,Xm,r(:,i));
    m0m = mus(i,:)-mu0;
    M = M0+Xm'*Xm+((beta0*nk(i))/beta(i))*(m0m'*m0m);     % equivalent to 10.62
    Sigmas(:,:,i) = M/v(i);
    U(:,:,i) = chol(M);
    logW(i) = -2*sum(log(diag(U(:,:,i))));      
end

model.alpha = alpha;
model.beta = beta;
model.mus = mus;
model.v = v;
model.U = U;
model.Sigmas = Sigmas;
model.logW = logW;

% Done
function model = expect(X, model)
alpha = model.alpha; % Dirichlet
beta = model.beta;   % Gaussian
mus = model.mus;         % Gasusian
v = model.v;         % Whishart
U = model.U;         % Whishart 
logW = model.logW;
n = size(X,1);
[k,d] = size(mus);

EQ = zeros(n,k);
for i = 1:k
    Q = U(:,:,i)'\bsxfun(@minus,X,mus(i,:))';
    EQ(:,i) = d/beta(i)+v(i)*dot(Q,Q,1);    % 10.64
end
ElogLambda = sum(psi(0,0.5*bsxfun(@minus,v+1,(1:d)')),1)+d*log(2)+logW; % 10.65
Elogpi = psi(0,alpha)-psi(0,sum(alpha)); % 10.66
logRho = -0.5*bsxfun(@minus,EQ,ElogLambda-d*log(2*pi)); % 10.46
logRho = bsxfun(@plus,logRho,Elogpi);   % 10.46
logR = bsxfun(@minus,logRho,logsumexp(logRho,2)); % 10.49
R = exp(logR);

model.logR = logR;
model.R = R;

% Done
function L = bound(X, model, prior)
alpha0 = prior.alpha;
beta0 = prior.beta;
v0 = prior.v;
logW0 = prior.logW;
alpha = model.alpha; 
beta = model.beta; 
v = model.v;         
logW = model.logW;
R = model.R;
logR = model.logR;
[n,d] = size(X);
k = size(R,2);

Epz = 0;
Eqz = dot(R(:),logR(:));
logCalpha0 = gammaln(k*alpha0)-k*gammaln(alpha0);
Eppi = logCalpha0;
logCalpha = gammaln(sum(alpha))-sum(gammaln(alpha));
Eqpi = logCalpha;
Epmu = 0.5*d*k*log(beta0);
Eqmu = 0.5*d*sum(log(beta));
logB0 = -0.5*v0*(logW0+d*log(2))-logMvGamma(0.5*v0,d);
EpLambda = k*logB0;
logB =  -0.5*v.*(logW+d*log(2))-logMvGamma(0.5*v,d);
EqLambda = sum(logB);
EpX = -0.5*d*n*log(2*pi);
L = Epz-Eqz+Eppi-Eqpi+Epmu-Eqmu+EpLambda-EqLambda+EpX;