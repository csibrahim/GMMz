function model = GMMz(X, K, options)
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


if nargin < 3
    
    options.tol = 1e-5;
    options.maxIter = 500;
    
end

O = ~isnan(X);

if(sum(~O(:))>0)
    model = GMMz_missing(X, K, options);
    return;
end


oldL = -inf;
model = init(X,K);
fprintf('Optimizing ... \n');
for iter = 1:options.maxIter
    
    model = expect(X,model);
    model = maximize(X,model);
    L = bound(X,model)/n;

    fprintf('%d\t%f\n',iter,L);
    
    if abs((L - oldL)/oldL) < options.tol; break; end
    
    oldL = L;
end

model = reduce(model);

function model = reduce(model)
    
    model.weights = model.alpha/sum(model.alpha);

    selected = floor(model.alpha)>3;

    K = sum(selected);


    model.K = K;
    model.alpha = model.alpha(selected);
    model.v = model.v(selected);
    model.beta = model.beta(selected);
    model.R = model.R(:,selected);
    model.logR = model.logR(:,selected);
    model.mus = model.mus(selected,:);
    model.Sigmas = model.Sigmas(:,:,selected);
    model.weights = model.weights(selected);
    model.M = model.M(:,:,selected);
    model.logW = model.logW(selected);
    
function model = init(X, K)

    n = size(X,1);

    if isstruct(K)  % init with a model
        model = K;

    elseif numel(K) == 1  % random init k

        
        fprintf('Initializing using k-means with 20 iterations ... \n');
        model = kmeans_missing(X,K,20);
        
        model.prior = setPriors(X);
        
        model = maximize(X,model);

    elseif all(size(K)==[1,n])  % init with labels
        label = K;
        k = max(label);
        model.R = full(sparse(1:n,label,1,n,k,n));
        model.prior = setPriors(X);
        model = maximize(X,model);
    else
        error('ERROR: init is not valid.');
    end
    

    


function prior = setPriors(X)
    
    [n,d] = size(X);
    
    init_model = kmeans_missing(X,1,5);

    prior.alpha = 1;
    prior.beta = 1;
    prior.mu = init_model.mus(1,:);
    prior.v = d+1;
    prior.M = (d+1)*init_model.Sigmas(:,:,1);

    prior.logW = -2*sum(log(diag(chol(prior.M))));

function model = maximize(X, model)

    alpha0 = model.prior.alpha;
    beta0 = model.prior.beta;
    mu0 = model.prior.mu;
    v0 = model.prior.v;
    M0 = model.prior.M;
    R = model.R;

    nk = sum(R); % 10.51
    alpha = alpha0+nk; % 10.58
    beta = beta0+nk; % 10.60
    v = v0+nk; % 10.63
    mus = bsxfun(@plus,R'*X,beta0*mu0);
    mus = bsxfun(@rdivide,mus,beta'); % 10.61
    
    [k,d] = size(mus);
    M = zeros(d,d,k);
    Sigmas = zeros(d,d,k);
    logW = zeros(1,k);
    
    X = [X;mu0];
    R = [R;beta0*ones(1,k)];
    
    for i = 1:k
        
        M(:,:,i) = M0+(R(:,i).*X)'*X-beta(i)*mus(i,:)'*mus(i,:);% equivalent to 10.62
        
        Sigmas(:,:,i) = M(:,:,i)/v(i);
        L = chol(M(:,:,i));
        logW(i) = -2*sum(log(diag(L)));      
    end

    model.alpha = alpha;
    model.beta = beta;
    model.mus = mus;
    model.v = v;
    model.M = M;
    model.Sigmas = Sigmas;
    model.logW = logW;

% Done
function model = expect(X, model)
    alpha = model.alpha; % Dirichlet
    beta = model.beta;   % Gaussian
    mus = model.mus;         % Gasusian
    v = model.v;         % Whishart
    M = model.M;         % Whishart 
    logW = model.logW;
    n = size(X,1);
    [k,d] = size(mus);

    EQ = zeros(n,k);
    
    for i = 1:k
        
        L = chol(M(:,:,i));
        delta = bsxfun(@minus,X,mus(i,:));
        Q = L'\delta';
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

function L = bound(X, model)
    alpha0 = model.prior.alpha;
    beta0 = model.prior.beta;
    v0 = model.prior.v;
    logW0 = model.prior.logW;
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