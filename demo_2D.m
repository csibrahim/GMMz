
%%%%%% CONFIGURATION %%%%%% 
rng(1);             % fix random seed
addpath VB_GMM/     % add path

% training options
maxIter = 100;      % maximum number of iterations
tol = 1e-10;        % early stoping criteria if no significant improvement is gained
init= 'kmeans';     % initialization option ('rand', 'conditional', 'kmeans')
cov = 'full';       % type of covariance ('full', 'diag')
display=1;          % display progress

% data creating options

n = 1000;           % number of samples
K = 100;            % number of mixtures (much higher than the real number of 3)
percentage = 0.0;   % percentage of data with a missing variable (half will be missing x and the other half will be missing y)

%%%%%% SETUP %%%%%% 

% create an artificial dataset
mean1 = [10 0];
Sigma1 = [10 0;0 1];

mean2 = [10 10];
Sigma2 = [5 -3;-3 3];

mean3 = [5 5];
Sigma3 = [2 0;0 2];


[group1,group2,group3] = sample(n,0.6,0.3,0.1); 

groups = [group1 group2 group3];

n = sum(groups);

X1 = mvnrnd(mean1,Sigma1,n(1));
X2 = mvnrnd(mean2,Sigma2,n(2));
X3 = mvnrnd(mean3,Sigma3,n(3));

% create missing variables

% remove the x from cluster 2
shuffle = randperm(n(2));
X2(shuffle(1:floor(percentage*n(2))),1) = nan;

% remove the y from cluster 3
shuffle = randperm(n(3));
X3(shuffle(1:floor(percentage*n(3))),2) = nan;

X = [X1;X2;X3];

missing = isnan(X);

d =  size(X,2);
%%%%%% TRAINING %%%%%% 

% training options
options.cyc=maxIter;
options.tol=tol;
options.init=init;
options.cov=cov;
options.display=display;

% Fitting a GMM to the data set
model = gmmvar_missing(X,K,options);
% model = kmeans_missing(X,K,100);

%%%%%%%%%%%%%%%%% PLOTING %%%%%%%%%%%%%%%%%%%%%%%%%%

% label axes
labels = ['x','y'];

% setting the nan values to the minimum-1 to puch the samples to the edges of the plot
X(missing(:,1),1) = min(X(~missing(:,1),1))-1;
X(missing(:,2),2) = min(X(~missing(:,2),2))-1;

hold on;

% plot samples with both x & y
both = ~missing(:,1)&~missing(:,2);
x_and_y = plot(X(both,1),X(both,2),'k.');

% plot samples with both x
missing_x = plot(X(missing(:,1),1),X(missing(:,1),2),'ko','MarkerFaceColor','g');

% plot samples with both y
missing_y = plot(X(missing(:,2),1),X(missing(:,2),2),'ko','MarkerFaceColor','y');

% plot the mixtures' means and covariances
for k=1:model.K
        
    ellipse = error_ellipse(model.Sigmas(:,:,k),model.mus(k,:),'style','-','conf',0.95);
    center = plot(model.mus(k,1),model.mus(k,2),'ko','markersize',10,'MarkerFaceColor','r');
    
end


if(sum(missing(:))>0)
    handles = [x_and_y(1),missing_x(1), missing_y(1), center(1), ellipse(1)];
    legends = {'x \& y',['Missing ',labels(1)],['Missing ',labels(2)],'Means', 'Covariances'};
    legend(handles,legends,'interpreter','latex');
else
    handles = [x_and_y(1),center(1), ellipse(1)];
    legends = {'x \& y','Means', 'Covariances'};
    legend(handles,legends,'interpreter','latex');
end
axis tight
xlabel('x');
ylabel('y');
title('Probability of x and y');


% plot the expectation of x given y and y given x
for o=1:2

    % rest the nan values for the missing data
    X(missing) = nan;
    u = 3-o;
    
    % creating a test set that covers the range of the observed variable
    Xs = nan(100,2);
    Xs(:,o) = linspace(min(X(:,o))-1,max(X(:,o)),100)';

    % predict the means, modes and medians of the unknoen variable 'u'
    % given the ovserved variable 'o'
    [mu, sigma, prob_mu, mode, prob_mode, median, prob_median] = predict_VB(Xs,o,u,model);

    % +\- 2 standard deviations to cover 95% of the range
    upper = mu+2*sqrt(sigma); 
    lower = mu-2*sqrt(sigma);

    % setting the nan values to the minimum-1 to push the samples to the edges of the plot
    X(missing(:,o),o) = min(Xs(:,o));
    X(missing(:,u),u) = min(lower)-1;

    figure;
    hold on;

    % plot the range +\- 2 std
    f = [upper; flip(lower)];
    pm2sigma = fill([Xs; flip(Xs)], f, [0.85 0.85 0.85]);

    
    x_and_y = plot(X(both,o),X(both,u),'k.');

    % plot samples with both variable 'o' observed
    missing_o = plot(X(missing(:,o),o),X(missing(:,o),u),'ko','MarkerFaceColor','g');
    
    % plot samples with both variable 'u' observed
    missing_u = plot(X(missing(:,u),o),X(missing(:,u),u),'ko','MarkerFaceColor','y');

    
    % plot the mode curve
    mode_plot = plot(Xs,mode,'b-','LineWidth',2);
    
    % plot the median curve
    median_plot = plot(Xs,median,'g-','LineWidth',2);
    
    % plot the mean curve
    mu_plot = plot(Xs,mu,'r-','LineWidth',2);
    
    
    if(sum(missing(:))>0)
        handles = [pm2sigma(1),mu_plot(1),mode_plot(1),median_plot(1),x_and_y(1),missing_o(1), missing_u(1)];
        legends = {'95\%', 'Mean', 'Mode','Median','x \& y',['Missing ',labels(o)],['Missing ',labels(u)]};
    else
        handles = [pm2sigma(1),mu_plot(1),mode_plot(1),median_plot(1),x_and_y(1)];
        legends = {'95\%', 'Mean', 'Mode','Median','x \& y'};
    end

    
    legend(handles,legends,'interpreter','latex');
    xlabel(labels(o));
    ylabel(labels(u));
    
    title([labels(u),' given ', labels(o)]);
    axis  tight
end

