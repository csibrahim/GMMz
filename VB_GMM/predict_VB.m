function [mu,sigma,prob_mu,mode,prob_mode, median, prob_median,values,PDF,CDF] = predict_VB(X,input,output,model,bins,range)
    
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
    
    
    
    for j=1:K 

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
    
    modes = zeros(n,K);
    
    for i=1:K
        for j=1:K
            Delta = (Euo_j(:,i)-Euo_j(:,j)).^2;
            modes(:,i) = modes(:,i)+exp(-0.5*Delta./Vuo_j(:,j)-0.5*log(Vuo_j(:,j))+log(Po_j(:,j)));
        end
    end
    
    [~, k] = max(modes,[],2);
   
    mode = full(sum(sparse(1:n,k,1,n,K).*Euo_j,2));

    mu = sum(Euo_j.*Po_j,2);
    sigma = sum((Vuo_j+Euo_j.^2).*Po_j,2)-mu.^2;

    prob_mu = zeros(n,1);
    prob_mode = zeros(n,1);

    for j=1:K
        Delta = (mu-Euo_j(:,j)).^2;
        prob_mu   = prob_mu   + exp( -0.5*(Delta./Vuo_j(:,j))-0.5*log(Vuo_j(:,j))-0.5*log(2*pi)+log(Po_j(:,j)));
        
        Delta = (mode-Euo_j(:,j)).^2;
        prob_mode = prob_mode + exp( -0.5*(Delta./Vuo_j(:,j))-0.5*log(Vuo_j(:,j))-0.5*log(2*pi)+log(Po_j(:,j)));
    end
    
    median = mu;
    prob_median = sum(exp(-0.5*((median-Euo_j).^2)./Vuo_j-0.5*log(Vuo_j)-0.5*log(2*pi)).*Po_j,2);
    
    
    min_y = mu-10*sqrt(sigma);
    max_y = mu+10*sqrt(sigma);
    
    tol = 1e-5;
    
    while(true)

        ZScore = bsxfun(@plus,-Euo_j,median)./sqrt(Vuo_j);
        CDF = sum(0.5*(1+erf(ZScore)).*Po_j,2);
        prob_median = sum(exp(-0.5*((median-Euo_j).^2)./Vuo_j-0.5*log(Vuo_j)-0.5*log(2*pi)).*Po_j,2);
        

        max_y(CDF>0.5) = median(CDF>0.5);
        min_y(CDF<0.5) = median(CDF<0.5);

        median = 0.5*(max_y-min_y)+min_y;

        error = max(abs(0.5-CDF));
        
        if(error<=tol)
            break;
        end

    end
    
    if(nargout>5)
        
        if(nargin>5)
            min_y = range(1);
            max_y = range(2);
            
        else
            min_y = mu-3*sqrt(sigma);
            max_y = mu+3*sqrt(sigma);
        end

        range = max_y-min_y;
        

        if(nargin<5)
            bins = 100;
        end
        
        values = ones(n,1)*(0:bins)/bins;
        values = bsxfun(@plus,bsxfun(@times,values,range),min_y);

        PDF = zeros(size(values));
        CDF = zeros(size(values));

        for j=1:K
            Delta = bsxfun(@minus,values,Euo_j(:,j));
            ZScore = bsxfun(@rdivide,Delta,sqrt(Vuo_j(:,j)));
            
            logPDF = bsxfun(@rdivide,Delta.^2,Vuo_j(:,j));
            logPDF = bsxfun(@plus,logPDF,log(Vuo_j(:,j)))+log(2*pi);
            PDF = PDF+exp(bsxfun(@plus,-0.5*logPDF,log(Po_j(:,j))));
            CDF = CDF+bsxfun(@times,0.5*(1+erf(ZScore)),Po_j(:,j));
        end
    end
    
end