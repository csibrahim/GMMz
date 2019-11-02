function [model, posterior] = kmeans_missing(X,K,maxIter,training,validation)

    if(nargin==5)
        X = X(training,:);
        training = true(sum(training),1);
    elseif(nargin==6)
        X = [X(training,:);X(validation,:)];
        training = [true(sum(training),1);false(sum(validation),1)];
    else
        training = true(size(X,1),1);
    end
        
    [n,d] = size(X);
    

    missing = isnan(X);
    O = ~missing;
    
    min_do = min(sum(O,2));
    
    list = true(n,1);
    groups = logical([]);
    
    while(sum(list)>0)
        first = find(list,1);
        group = false(n,1);
        group(list) = sum(abs(bsxfun(@minus,missing(list,:),missing(first,:))),2)==0;
        groups = [groups group];
        list(group)=false;
    end
    
    X(missing) = 0;

    XX = X(training,:)'*X(training,:);
    OO = O(training,:)'*O(training,:);
    
    XX = XX./OO;
    M = (X(training,:)'*O)./OO;
    
    Sigma = XX-M.*M';
    
    Sigma(OO==0) = 0;
    
    mu = diag(M)';
    mu = (mu.*sum(O))./(sum(O)+1);
    
    shuffle = randperm(sum(training));
    training_ids = find(training);
    training_shuffled = training_ids(shuffle);
    
    mus = X(training_shuffled(1:K),:);
    
    
    logLikelihood = zeros(n,K); 

    max_mll = -inf;
    old_mll = 0;
    
    
    for iter = 1:maxIter

        for j=1:K

            for g=1:size(groups,2)

                group = groups(:,g);

                o = ~missing(find(group,1),:)&~isnan(mus(j,:));
                
                do = sum(o);
            

                Delta = X(group,o)-mus(j,o);
                
                logLikelihood(group, j) = -0.5*sum(Delta.^2,2)-0.5*(do-min_do)*log(2*pi);
            end

        end
        
        [~,cids] = max(logLikelihood,[],2);
        
        posterior = sparse(1:n,cids,1,n,K);
        
        likelihood = exp(logLikelihood)+eps;
        
        Px = sum(likelihood,2)*exp(-(min_do/2)*log(2*pi)); 
        
        gravity = 1e-2;
        WX = posterior(training,:)'*X(training,:)+gravity*mu;
        WO = posterior(training,:)'*O(training,:)+gravity;
        
        mus = WX./WO;
        
        
        train_mll = mean(log(Px(training)));
        
        if(nargin>4)
        
            valid_mll = mean(log(Px(~training)));
            
            if(valid_mll>=max_mll)
                fprintf('%d\t%1.4e\t[%1.4e]\n', iter,train_mll,valid_mll);
                max_mll = valid_mll;
            else
                fprintf('%d\t%1.4e\t %1.4e\n', iter,train_mll,valid_mll);
            end
        else
            fprintf('%d\t%15.4e\n', iter,mean(log(Px(training))));
        end
        
        if(abs((train_mll - old_mll)/old_mll*100) < 1e-10)
            break; 
        else
            old_mll=train_mll;
        end
        

    end
    
    
    Nm = full(sum(posterior(training,:)));
    
    remove = Nm==0;
    
    Nm(remove) = [];
    mus(remove,:) = [];
    posterior = full(posterior(:,~remove));
    
    K = K-sum(remove);
    
    Sigmas = zeros(d,d,K);
    
    weights = Nm/sum(Nm);
    
    for j=1:K
            
        Sigmas(:,:,j) = Sigma/nthroot(K,d);
        
    end
    
    posterior = full(posterior);
    model.mu = mu;
    model.Sigma = Sigma;
    
    model.K = K;
    model.mus = mus;
    model.Sigmas = Sigmas;
    model.weights = weights;
     
end