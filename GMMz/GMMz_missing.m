function model = GMMz_missing(X, K, options)
% Variational Bayesian inference for Gaussian mixture.
% Input: 
%   X: d x n data matrix
%   m: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or model structure
% Output:
%   model: trained model structure
%   L: variational lower bound
% Reference: Pattern Recognition and Machine Learning by Christopher M. Bishop (P.474)
% Written by Mo Chen (sth4nth@gmail.com).

fprintf('Variational Bayesian Gaussian mixture: running ... \n');
[n,d] = size(X);

O = ~isnan(X);

list = true(n,1);
groups = logical([]);

while(sum(list)>0)
  first = find(list,1);
  group = false(n,1);
  group(list) = sum(abs(bsxfun(@minus,O(list,:),O(first,:))),2)==0;
  groups = [groups group];
  list(group)=false;
end

if nargin < 3
    
    options.tol = 1e-5;
    options.maxIter = 500;
    
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
    model = expect(X,model,groups);
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

    [n,d] = size(X);

    O = ~isnan(X);
    X(~O) = 0;

    alpha0 = prior.alpha;
    beta0 = prior.beta;
    mu0 = prior.mu;
    v0 = prior.v;
    M0 = prior.M;
    R = model.R;

    R = 1.5*R;

    nk = sum(R); % 10.51
    alpha = alpha0+nk; % 10.58
    beta = beta0+nk; % 10.60
    v = v0+nk; % 10.63

    RO = R'*O;

    xk = (R'*X)./(RO);
    xk(RO==0) = 0;

    mus = (xk.*nk'+beta0*mu0)./beta';% 10.61

    k = size(mus,1);

    M = zeros(d,d,k);
    Sigmas = zeros(d,d,k);
    logW = zeros(1,k);

    for i = 1:k

        wX = R(:,i).*X;
        wO = R(:,i).*O;
        XwX = wX'*X;
        OwO = wO'*O;
        XwO = wX'*O;


        Sk = XwX-(XwO.*XwO')./OwO;

        Sk = Sk./OwO;
        Sk(OwO==0) = 0;

        m0mTm0m = (xk(i,:)-mu0)'*(xk(i,:)-mu0);

        M(:,:,i) = M0+nk(i)*(Sk+beta0*(m0mTm0m)/beta(i));     % equivalent to 10.62


        [L,p] = chol(M(:,:,i));

        if(p~=0)
            M(:,:,i) = diag(diag(M(:,:,i)));
            logW(i) = -sum(log(diag(M(:,:,i))));
        else
            logW(i) = -2*sum(log(diag(L)));
        end

        Sigmas(:,:,i) = M(:,:,i)/v(i);

    end

    model.alpha = alpha;
    model.beta = beta;
    model.mus = mus;
    model.v = v;
    model.M = M;
    model.logW = logW;
    model.Sigmas = Sigmas;

% Done
function model = expect(X, model,groups)


    alpha = model.alpha; % Dirichlet
    beta = model.beta;   % Gaussian
    mus = model.mus;     % Gasusian
    v = model.v;         % Whishart
    M = model.M;         % Whishart 
    n = size(X,1);
    [k,d] = size(mus);


    O = ~isnan(X);


    logRho = zeros(n,k);

    for i = 1:k

        for g=1:size(groups,2)

            group = groups(:,g);

            o = O(find(group,1),:);

            do = sum(o);

            U = chol(M(o,o,i));
            logW = -2*sum(log(diag(U)));

            Q = U'\bsxfun(@minus,X(group,o),mus(i,o))';
            EQ = do/beta(i)+v(i)*dot(Q,Q,1);    % 10.64

            ElogLambda = sum(psi(0,0.5*bsxfun(@minus,v(i)+1,(1:do)')),1)+do*log(2)+logW; % 10.65
            Elogpi = psi(0,alpha(i))-psi(0,sum(alpha)); % 10.66

            logRho(group,i) = -0.5*bsxfun(@minus,EQ,ElogLambda-do*log(2*pi)); % 10.46
            logRho(group,i) = bsxfun(@plus,logRho(group,i),Elogpi);   % 10.46
        end
    end

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