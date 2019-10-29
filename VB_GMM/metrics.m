function scores = metrics(y,mu,prob,fun)
    
    
    n = length(y);
    [~,order] = sort(-prob);
    
    y = y(order);
    prob = prob(order);
    mu = mu(order);

    scores = cumsum(fun(y,mu,prob))./(1:n)';

end

