function [model, posterior] = kmeans_missing(X,K,maxIter,training,validation)

    gravity = 0.01;
    
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
    observed = ~missing;
    
    min_do = min(sum(observed,2));
    
    list = true(n,1);
    groups = logical([]);
    
    while(sum(list)>0)
        first = find(list,1);
        group = false(n,1);
        group(list) = sum(abs(bsxfun(@minus,missing(list,:),missing(first,:))),2)==0;
        groups = [groups group];
        list(group)=false;
    end
    
    
    Sigma = zeros(d,d);
    mu = zeros(1,d);

    for x=1:d

        group = observed(:,x)&training;

        mu(x) = sum(X(group,x))/(sum(group)+gravity);

        Delta = X(group,x)-mu(x);
        Sigma(x,x) = (sum(Delta.^2)+gravity)/(sum(group)+gravity);

        for y=x+1:d

            group = observed(:,x)&observed(:,y)&training;

            if(sum(group)>0)

                mu_x = mean(X(group,x));
                mu_y = mean(X(group,y));

                Delta_x = X(group,x)-mu_x;
                Delta_y = X(group,y)-mu_y;

                Sigma(x,y) = sum(Delta_x.*Delta_y)/(sum(group)+gravity);
                Sigma(y,x) = Sigma(x,y);
            end

        end
    end
    
    shuffle = randperm(sum(training));
    training_ids = find(training);
    training_shuffled = training_ids(shuffle);
    
    mus = X(training_shuffled(1:K),:);
    
    X(missing) = 0;
    
    logLikelihood = zeros(n,K); 

    max_mll = -inf;
    
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
        
        mus = (posterior(training,:)'*X(training,:)+gravity*mu)./(posterior(training,:)'*observed(training,:)+gravity);
        
        if(nargin>4)
            
            train_mll = mean(log(Px(training)));
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
        
        

    end
    
    Nm = full(sum(posterior(training,:)));
    
    remove = Nm==0;
    
    Nm(remove) = [];
    mus(remove,:) = [];
    posterior(:,remove) = [];
    
    K = K-sum(remove);
    
    Sigmas = zeros(d,d,K);
    
    weights = Nm/sum(Nm);
    
    
    diagXX = (0.1*diag(Sigma)'+mu.^2);
    diags = (posterior(training,:)'*X(training,:).^2+gravity*diagXX)./(posterior(training,:)'*observed(training,:)+gravity)-mus.^2;
        
    for j=1:K

        Sigmas(:,:,j) = diag(diags(j,:));

    end
    
    
    posterior = full(posterior);
    model.mu = mu;
    model.Sigma = Sigma;
    
    model.K = K;
    model.mus = mus;
    model.Sigmas = Sigmas;
    model.weights = weights;
     
end