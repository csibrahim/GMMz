function [model,FrEn,U] = gmmvar_missing(X,K,options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   [gmm,FrEn] = gmmvar (data,K,options)
%
%   Computes the Aposteriori pdfs of a mixture model using the variational
%   Bayesian learning approach for a given number of classes, K. K may
%   be a vector - a model will be fitted for each element of
%   K. Convergence is assumed after loopmax (default 100) iterations. 
%   
%   Function returns the mixture components' mean and co-variance
%   matrices in mue and sigma, as well as the class  posterior
%   membership probability in pjgx. FrEn is the  estimated free energy
%   for each model 
%
%   The structure data consists of 
%       data           the actual data for clustering 
%
%
%  options may be set to contain:
%     options.cyc :=  max. number of iterations (default=50)
%     options.tol :=  minimum improvement of free energy
%                     function (default=1e-5%)
%     options.init := initial fitting: 
%             conditional : full conditioanl (default) 
%             rand        : random posteriors
%             kmeans      : k-means init
%     options.cov := shape of covariance matrix: 
%                    'diag'=diagonal ; 'full'=full cov. mat. default)
%
%  options.display :=  display intermeidate free energy values: 
%                      1=yes 0=no (dafault)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

default.cyc=500;
default.tol=1e-5;
default.init='rand';
default.cov='full';
default.min_cov=eps;

[n,d] = size(X);

if (nargin<1) 
  disp('Usage: gmm=gmmvar(data,K,options);');
  return;
end

if nargin<2
  error('Missing Parameter for Number of Kernels ');
end

if (n<d)
  X=X';
  [n,d]=size(X);
end

if ((nargin<3) || isempty(options))
  options=default;
else
  if ~isfield(options,'cyc')
    options.cyc = default.cyc;
  end

  if ~isfield(options,'tol')
    options.tol = default.tol;
  end
  
  if isfield(options,'cov')
    if ~ismember(options.cov,['diag','full','spherical'])
      error('Unknown covariance matrix shape option');
    end
  else
    options.cov=default.cov;
  end

  if isfield(options,'init')
    if ~ismember(options.init,['conditional','rand','kmeans'])
      error('Unknown intialisation option');
    elseif strcmp(options.init,'kmeans')
      if any(K==1)
	disp('Changing intialisation to default');
	options.init='conditional';
      elseif any(d>K)
        fprint('K-means initalisation not recommended %d>%d',d,K);
      end
    end
  else
    options.init=default.init;
  end
  
  if isfield(options,'display')
    if ~ismember(options.display,[0 1])
      error('Unknown display flag');
    end
  else
    options.display=default.display;
  end
  
  
end

oldFrEn=1;
FrEn=0;
U=nan(options.cyc,3);

% initialise
gmm = gmmvarinit_missing(X,K,options);
K = gmm.K;

missing = isnan(X);

list = true(n,1);
groups = logical([]);

while(sum(list)>0)
  first = find(list,1);
  group = false(n,1);
  group(list) = sum(abs(bsxfun(@minus,missing(list,:),missing(first,:))),2)==0;
  groups = [groups group];
  list(group)=false;
end

% gmm.pjgx=estep(gmm,X,K,groups);

for cyc=1:options.cyc

    % The E-Step, i.e. estimating Q(hidden variables)
    gmm.pjgx=estep(gmm,X,K,groups);

    % The M-Step, i.e. estimating Q(model parameters)
    gmm=mstep(gmm,X,K,groups);

    % computation of free energy 
    
    [FrEn,U(cyc,1:3)]=freeener(gmm,X,K,groups);

    if(abs((FrEn - oldFrEn)/oldFrEn*100) < options.tol)
     break; 
    else
      oldFrEn=FrEn;
    end

    if(options.display)
      fprintf('Iteration %d ; Free-Energy = %f\n',cyc,FrEn); 
    end


end

fprintf('Model: %d kernels, %d dimensions, %d data samples\n',K,d,n);
fprintf('Final Free-Energy (after %d iterations)  = %f\n',cyc,FrEn);


clear Dir_alpha
[Dir_alpha{1:K}]=deal(gmm.post.Dir_alpha);
Dir_alpha=cat(2,Dir_alpha{:});

Nk = floor(Dir_alpha-1);

selected = Nk>2;

mus = zeros(K,d);
Sigmas = zeros(d,d,K);

weights = Dir_alpha(selected);
weights = weights/sum(weights);

counter = 0;
for k=1:K
    
    Nk = floor(gmm.post(k).Dir_alpha - 1);
    
    if (Nk > 2) % the wishart is singular otherwise
        
        counter = counter+1;
        
        mus(counter,:) = gmm.post(k).Norm_Mu;
        Sigmas(:,:,counter) = gmm.post(k).Wish_B./gmm.post(k).Wish_alpha;
    end
end

model.K = sum(selected);
model.mus = mus;
model.Sigmas = Sigmas;
model.weights = weights;

return

%%%%%%%%%%%%%%%%%%%%%%%%%%  E-STEP  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function pjgx = estep(gmm,X,K, groups)


observed = ~isnan(X);
min_do = min(sum(observed,2));

% PsiDiralphasum is removed because it will cancel out during normalization

% [Dir_alpha{1:K}]=deal(gmm.post.Dir_alpha);
% Dir_alpha=cat(2,Dir_alpha{:});
% PsiDiralphasum=psi(sum(Dir_alpha));

for k=1:K

    qp=gmm.post(k);
    
    for g=1:size(groups,2)
        
        group = groups(:,g);

        o = observed(find(group,1),:);

        do = sum(o);
        
        log2pi = 0.5*(do-min_do)*log(2*pi);

        L = chol(qp.Wish_B(o,o));
        Li = inv(L);
        Wish_iBoo = Li*Li';
        
        ldetWishB=sum(log(diag(L)));
        
        delta = X(group,o)-qp.Norm_Mu(o)';
        dist = -0.5*sum((delta/L).^2,2)*qp.Wish_alpha;
        
        
        PsiDiralpha=psi(qp.Dir_alpha);
        PsiWish_alphasum = 0.5*sum(psi(qp.Wish_alpha+0.5-(1:do)/2));

        NormWishtrace=0.5*qp.Wish_alpha*sum(sum((Wish_iBoo.*qp.Norm_Cov(o,o))));

        gmm.pjgx(group,k)=exp(PsiDiralpha+PsiWish_alphasum-NormWishtrace+dist-ldetWishB-log2pi);
    end

end


% normalise posteriors of hidden variables.
gmm.pjgx=gmm.pjgx+eps;
px=sum(gmm.pjgx,2);
gmm.pjgx=bsxfun(@rdivide,gmm.pjgx,px);
pjgx=gmm.pjgx;

% another way of normalising
%    for k=1:K(a),
%      col_sum=gmm(a).pjgx(:,k)*ones(1,K(a));
%      inv_prob=sum(gmm(a).pjgx./col_sum,2);
%      if any(inv_prob==0)
% 	disp(['Zero normalisation constant for hidden variable' ...
% 	      ' posteriors']);
% 	return;
%      else
% 	gmm(a).pjgx(:,k)=1./sum(gmm(a).pjgx./col_sum,2);
%      end;
%    end ;

return;					% estep
 
%%%%%%%%%%%%%%%%%%%%%%%%%%  M-STEP  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [gmm]=mstep(gmm,X,K, groups)

[~,d] = size(X);

observed = ~isnan(X);

pr=gmm.priors;			% model priors

gammasum=sum(gmm.pjgx,1);

for k = 1:K
  
    qp=gmm.post(k);
    
    postprec = gammasum(k)*qp.Wish_alpha*qp.Wish_iB+pr.Norm_Prec;
    postvar=inv(postprec);
    
    weidata = zeros(d,1);
    w = zeros(d,1);
    
    for g=1:size(groups,2)
        
        group = groups(:,g);

        o = observed(find(group,1),:);

        weidata(o) = weidata(o)+X(group,o)'*gmm.pjgx(group,k);
        w(o) = w(o)+sum(gmm.pjgx(group,k));
    end
    
    weidata = (weidata./w)*gammasum(k);
    
    Norm_Mu = postvar*(qp.Wish_alpha*qp.Wish_iB*weidata+pr.Norm_Prec*pr.Norm_Mu);
    Norm_Prec=postprec;
    Norm_Cov=postvar;

    Wish_alpha=0.5*gammasum(k)+pr.Wish_alpha;
    
    sampvar = zeros(d);
    
    for x=1:d
                
        group = observed(:,x);

        Delta = X(group,x)-Norm_Mu(x);
        sampvar(x,x) = gmm.pjgx(group,k)'*(Delta.^2)/sum(gmm.pjgx(group,k));
        
        for y=x+1:d

            group = observed(:,x)&observed(:,y);

            if(sum(group)>0)

                Delta_x = X(group,x)-Norm_Mu(x);
                Delta_y = X(group,y)-Norm_Mu(y);

                sampvar(x,y) = bsxfun(@times,Delta_x,gmm.pjgx(group,k))'*Delta_y/sum(gmm.pjgx(group,k));
                
                sampvar(y,x) = sampvar(x,y);
            end

        end
    end

    Wish_B=0.5*gammasum(k)*(sampvar+Norm_Cov)+pr.Wish_B;
    Wish_iB=inv(Wish_B);

    % Update posterior Dirichlet
    Dir_alpha=gammasum(k)+pr.Dir_alpha(k);

    gmm.post(k).Norm_Mu=Norm_Mu;
    gmm.post(k).Norm_Prec=Norm_Prec;
    gmm.post(k).Norm_Cov=Norm_Cov;
    gmm.post(k).Wish_alpha=Wish_alpha;
    gmm.post(k).Wish_B=Wish_B;
    gmm.post(k).Wish_iB=Wish_iB;
    gmm.post(k).Dir_alpha=Dir_alpha;
end

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%  FREE ENERGY  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [FrEn,U]=freeener(gmm,X,K, groups)

observed = ~isnan(X);

KLdiv=0;
avLL=0;
pr=gmm.priors;

[Dir_alpha{1:K}]=deal(gmm.post.Dir_alpha);
Dir_alpha=cat(2,Dir_alpha{:});
Dir_alphasum=sum(Dir_alpha);
PsiDir_alphasum=psi(Dir_alphasum);

% entropy of hidden variables, which are not zero
pjgx=gmm.pjgx(:);
ndx=find(pjgx~=0);
Entr=sum(sum(pjgx(ndx).*log(pjgx(ndx))));

for k=1:K
  
    qp=gmm.post(k);

    PsiDiralpha=psi(qp.Dir_alpha);
    
    for g=1:size(groups,2)
        
        group = groups(:,g);

        o = observed(find(group,1),:);

        do = sum(o);
        
        log2pi = 0.5*do*log(2*pi);
        
        L = chol(qp.Wish_B(o,o));
        Li = inv(L);
        Wish_iBoo = Li*Li';
        ldetWishB=sum(log(diag(L)));
        
        delta = X(group,o)-qp.Norm_Mu(o)';
        dist = -0.5*sum((delta/L).^2,2)*qp.Wish_alpha;
       
        NormWishtrace=0.5*qp.Wish_alpha*sum(sum((Wish_iBoo'.*qp.Norm_Cov(o,o))));
        PsiWish_alphasum = 0.5*sum(psi(qp.Wish_alpha+0.5-(1:do)/2));

        avLL=avLL+sum(gmm.pjgx(group,k)).*(PsiDiralpha-PsiDir_alphasum-ldetWishB+PsiWish_alphasum-NormWishtrace-log2pi)+gmm.pjgx(group,k)'*dist;

    end
    
    VarDiv=wishart_kl(qp.Wish_B,pr.Wish_B,qp.Wish_alpha,pr.Wish_alpha);
    MeanDiv=gauss_kl(qp.Norm_Mu,pr.Norm_Mu,qp.Norm_Cov,pr.Norm_Cov);
    KLdiv=KLdiv+VarDiv+MeanDiv;
    
end

% KL divergence of Dirichlet

KLdiv=KLdiv+dirichlet_kl(Dir_alpha,pr.Dir_alpha);

FrEn=Entr-avLL+KLdiv;
U=[Entr -avLL +KLdiv];

return

