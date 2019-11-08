function values = percentile(X,input,output,model, prct)
    
    n = size(X,1);
    
    missing = isnan(X);
    
    X(missing) = 0;
    
    list = true(n,1);
    groups = logical([]);
    
    while(sum(list)>0)
        first = find(list,1);
        group = false(n,1);
        group(list) = sum(abs(bsxfun(@minus,missing(list,:),missing(first,:))),2)==0;
        groups = [groups group];
        list(group)=false;
    end
    
    
    K = model.K;
    weights = model.weights;
    mus = model.mus;
    Sigmas = model.Sigmas;
    
    Euo_j = zeros(n,K);
    Vuo_j = zeros(n,K);
    Po_j = zeros(n,K);
    
    
    
    for j=1:K %  for each data point

        for g=1:size(groups,2)

            group = groups(:,g);

            o = ~missing(find(group,1),input);

            do = sum(o);


            Delta = X(group,input(o))-mus(j,input(o));
            
            iSooSu = Sigmas(input(o),input(o),j)\Sigmas(input(o),output,j);
            
            Euo_j(group,j) = mus(j,output)+Delta*iSooSu;
            Vuo_j(group,j) = Sigmas(output,output,j)-Sigmas(output,input(o),j)*iSooSu;
            
            

            L = chol(Sigmas(input(o),input(o),j));
            half_logDetS = sum(log(diag(L)));
            DeltaLinv = Delta/L;

            Po_j(group, j) = exp(-0.5*sum(DeltaLinv.^2,2)-half_logDetS-(do/2)*log(2*pi)+log(weights(j)))+1e-15;
        end

    end
    
    Po_j = bsxfun(@rdivide,Po_j,sum(Po_j,2));
    
    
    mu = sum(Euo_j.*Po_j,2);
    sigma = sum((Vuo_j+Euo_j.^2).*Po_j,2)-mu.^2;

    
    

    values = repmat(mu,1,length(prct));
    
    tol = 1e-5;
    
    for i=1:length(prct)
        
        
        error = 1;
        
        min_y = mu-10*sqrt(sigma);
        max_y = mu+10*sqrt(sigma);
        
        while(error>tol)
            
            ZScore = bsxfun(@plus,-Euo_j,values(:,i))./sqrt(2*Vuo_j);
            CDF = sum(0.5*(1+erf(ZScore)).*Po_j,2);

            max_y(CDF>prct(i)) = values(CDF>prct(i),i);
            min_y(CDF<prct(i)) = values(CDF<prct(i),i);

            values(:,i) = 0.5*(max_y-min_y)+min_y;

            error = max(abs(prct(i)-CDF));

        end
        
    end
    
    
    
end