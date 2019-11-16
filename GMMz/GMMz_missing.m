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

oldL = -inf;
model = init(X,K);
fprintf('Optimizing ... \n');
for iter = 1:options.maxIter
    
    model = expect(X,model,groups);
    model = maximize(X,model);
    L = bound(X,model)/n;

    fprintf('%d\t%f\n',iter,L);
    
    if abs((L - oldL)/oldL) < options.tol; break; end
    
    oldL = L;
    
    model = reduce(model);
end

model = reduce(model);

function model = reduce(model)
    
    model.weights = model.alpha/sum(model.alpha);

    selected = floor(model.alpha)>3;

    K = sum(selected);


    model.K = K;
    model.alpha = model.alpha(selected);
    model.v = model.v(selected);
    model.beta = model.beta(selected,:);
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

    [n,d] = size(X);

    O = ~isnan(X);
    
    X(~O) = 0;

    alpha0 = model.prior.alpha;
    beta0 = model.prior.beta;
    mu0 = model.prior.mu;
    v0 = model.prior.v;
    M0 = model.prior.M;
    R = model.R;

    RO = R'*O;
    
    nk = sum(R); % 10.51
    alpha = alpha0+nk; % 10.58
    beta = RO+beta0; % 10.60
    v = v0+nk; % 10.63

    mus = (R'*X+beta0*mu0)./beta;% 10.61

    k = size(mus,1);

    M = zeros(d,d,k);
    Sigmas = zeros(d,d,k);
    logW = zeros(1,k);

    X = [X;mu0];
    O = [O;true(1,d)];
    
    for i = 1:k

        
        RXi = [R(:,i);beta0].*X;
        ROi = [R(:,i);beta0].*O;
        
        XRX = X'*RXi;
        ORO = O'*ROi;
        OXR = O'*RXi;

        MM = (OXR.*OXR')./ORO;
        
        
        Mk = XRX-MM;
        
        Mk = (nk(i)+beta0)*(Mk./ORO);
        
        M(:,:,i) = M0+Mk;
       
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

function model = expect(X, model,groups)


    alpha = model.alpha; % Dirichlet
    beta = model.beta;   % Gaussian
    mus = model.mus;     % Gasusian
    v = model.v;         % Whishart
    M = model.M;         % Whishart 
    n = size(X,1);
    k = size(mus,1);


    O = ~isnan(X);


    logRho = zeros(n,k);

    for i = 1:k
        
        for g=1:size(groups,2)

            group = groups(:,g);

            o = O(find(group,1),:);

            do = sum(o);

            U = chol(M(o,o,i));
            logW = -2*sum(log(diag(U)));

            delta = bsxfun(@minus,X(group,o),mus(i,o));
            Q = U'\delta';
            
            EQ = sum(1./beta(i,o))+v(i)*dot(Q,Q,1);    % 10.64
            
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
    Eqmu = 0.5*sum(log(beta(:)));
    logB0 = -0.5*v0*(logW0+d*log(2))-logMvGamma(0.5*v0,d);
    EpLambda = k*logB0;
    logB =  -0.5*v.*(logW+d*log(2))-logMvGamma(0.5*v,d);
    EqLambda = sum(logB);
    EpX = -0.5*d*n*log(2*pi);
    L = Epz-Eqz+Eppi-Eqpi+Epmu-Eqmu+EpLambda-EqLambda+EpX;