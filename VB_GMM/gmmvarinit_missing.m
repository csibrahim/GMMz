function [gmm] = gmmvarinit_missing(X,K,options)
% initialises the gaussian mixture model for Variational GMM algorithm
  
  gmm.K=K;
  [n,d]=size(X);
    
    
  midscale = median(X,'omitnan')';
  drange = range(X)';	
  % educated guess with scaling
  
  % define P-priors
  defgmmpriors=struct('Dir_alpha',[],'Norm_Mu',[],'Norm_Cov', ...
		      [],'Norm_Prec',[],'Wish_B',[],'Wish_iB',[],...
		      'Wish_alpha',[],'Wish_k',[]);
  
    
  defgmmpriors.Dir_alpha=ones(1,K);
  defgmmpriors.Norm_Mu=midscale;
  defgmmpriors.Norm_Cov=diag(drange.^2);
  defgmmpriors.Norm_Prec=inv(defgmmpriors.Norm_Cov);
  defgmmpriors.Wish_B=diag(drange);
  defgmmpriors.Wish_iB=inv(defgmmpriors.Wish_B);
  defgmmpriors.Wish_alpha=d+1;
  defgmmpriors.Wish_k=d;
 
  
  % assigning default P-priors 
  if ~isfield(options,'priors')
    gmm.priors=defgmmpriors;
  else
    % priors not specified are set to default
    gmmpriorlist=fieldnames(defgmmpriors);
    fldname=fieldnames(gmm.priors);
    misfldname=find(~ismember(gmmpriorlist,fldname));
    
    for i=1:length(misfldname)
      priorval = getfield(defgmmpriors,gmmpriorlist{i});
      gmm.priors = setfield(gmm.priors,gmmpriorlist{i},priorval);
    end
    
  end

    
  % initialise posteriors
  switch options.init
   case 'rand'
    % sample mean from data
    
    km_model = kmeans_missing(X,K,1);
    
    K = km_model.K;
    gmm.K = K;
    gmm.priors.Dir_alpha = ones(1,K);
    
    Nm = 3+n*ones(1,K)/K;
    
    Covcentre = cov(km_model.mus);
    
    for k=1:K
        
        gmm.post(k).Norm_Mu = km_model.mus(k,:)';
        gmm.post(k).Norm_Cov=Covcentre;
        gmm.post(k).Norm_Prec = inv(Covcentre);
        gmm.post(k).Wish_alpha=Nm(k);
        gmm.post(k).Wish_B=(km_model.Sigma/nthroot(K,d))*Nm(k);
        gmm.post(k).Wish_iB=inv(gmm.post(k).Wish_B);
        gmm.post(k).L = chol(gmm.post(k).Wish_B);
        gmm.post(k).Li = chol(gmm.post(k).Wish_B);
        gmm.post(k).Dir_alpha=Nm(k);
    end
    
   case 'conditional'
       
    fprintf('initializing using k-means with 20 iterations\n');
    [km_model, Pcgx] = kmeans_missing(X,K,20);

    
    K = km_model.K;
    gmm.K = K;
    gmm.priors.Dir_alpha = ones(1,K);
    
    Nm = sum(Pcgx);
    
    for k=1:K
        Cov=km_model.Sigmas(:,:,k);
        Prec=inv(Cov);
        
        gmm.post(k).Norm_Prec=Nm(k)*Prec+gmm.priors.Norm_Prec;
        gmm.post(k).Norm_Cov=inv(gmm.post(k).Norm_Prec);
        gmm.post(k).Norm_Mu=gmm.post(k).Norm_Cov*(Nm(k)*Prec*km_model.mus(k,:)'+gmm.priors.Norm_Prec*gmm.priors.Norm_Mu);
        
        gmm.post(k).Wish_alpha=0.5*(gmm.priors.Wish_alpha+Nm(k));
        gmm.post(k).Wish_B=gmm.priors.Wish_B+Nm(k)*Cov;
        gmm.post(k).Wish_iB=inv(gmm.post(k).Wish_B);
        
        gmm.post(k).Dir_alpha=gmm.priors.Dir_alpha(k)+Nm(k);
        
    end

   case 'kmeans'
    
    fprintf('initializing using k-means with 20 iterations\n')
    [km_model, Pcgx] = kmeans_missing(X,K,20);
    
    K = km_model.K;
    gmm.K = K;
    gmm.priors.Dir_alpha = ones(1,K);
    
    Covcentre = cov(km_model.mus);
    
    alphas = sum(Pcgx)+3;

    for k=1:K
        
        gmm.post(k).Norm_Mu = km_model.mus(k,:)';
        gmm.post(k).Norm_Cov = Covcentre;
        gmm.post(k).Norm_Prec = inv(Covcentre);
        gmm.post(k).Wish_alpha = alphas(k);
        gmm.post(k).Wish_B = km_model.Sigmas(:,:,k)*alphas(k);
        gmm.post(k).Wish_iB = inv(gmm.post(k).Wish_B);
        gmm.post(k).Dir_alpha = alphas(k);
    end
       
   otherwise
    error('Unknown intialisation option');
  end
 
return
