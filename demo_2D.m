
%%%%%%%%%%%%%%%%% CONFIGURATION %%%%%%%%%%%%%%%%%%

rng(1);                     % fix random seed
addpath GMMz/               % add path

% training options
maxIter = 1000;             % maximum number of iterations
tol = 1e-10;                % early stoping criteria if no significant improvement is gained

% data creating options

n = 1000;                   % number of samples
K = 100;                    % number of mixtures (much higher than the real number of 3)
percentage = 0.5;           % percentage of data with a missing variable (half will be missing x and the other half will be missing y)

%%%%%%%%%%%%%%%%% SETUP %%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%

% training options
options.maxIter=maxIter;
options.tol=tol;

% Fitting a GMM to the data set
model = GMMz(X,K,options);

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

handles = [];
legends = {};

if(sum(both)>0)
    handles = x_and_y(1);
    legends = {'x \& y'};
end

if(sum(missing(:,1))>0)
    handles = [handles,missing_x(1)];
    legends = {legends{:},['Missing ',labels(1)]};
end

if(sum(missing(:,2))>0)
    handles = [handles, missing_y(1)];
    legends = {legends{:},['Missing ',labels(2)]};
end

handles = [handles,center(1), ellipse(1)];
legends = {legends{:},'Means', 'Covariances'};

legend(handles,legends,'interpreter','latex');

axis tight
xlabel('x');
ylabel('y');
title('Probability of x and y');

% plot the expectation of x given y and y given x
for u=1:2

    % rest the nan values for the missing data
    X(missing) = nan;
    o = 3-u;
    
    bins = 1000;
    % creating a test set that covers the range of the observed variable
    Xs = nan(bins,2);
    range = max(X(:,o))-min(X(:,o));
    Xs(:,o) = linspace(min(X(:,o))-range/10,max(X(:,o)+range/10),bins)';

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

    
    % plot samples with both variables observed
    x_and_y = plot(X(both,o),X(both,u),'k.');

    % plot samples with only variable 'o' observed
    missing_o = plot(X(missing(:,o),o),X(missing(:,o),u),'ko','MarkerFaceColor','g');
    
    % plot samples with only variable 'u' observed
    missing_u = plot(X(missing(:,u),o),X(missing(:,u),u),'ko','MarkerFaceColor','y');

    
    % plot the mode curve
    mode_plot = plot(Xs,mode,'b-','LineWidth',2);
    
    % plot the median curve
    median_plot = plot(Xs,median,'g-','LineWidth',2);
    
    % plot the mean curve
    mu_plot = plot(Xs,mu,'r-','LineWidth',2);
    
    handles = [pm2sigma(1),mu_plot(1),mode_plot(1),median_plot(1)];
    legends = {'95\%', 'Mean', 'Mode','Median'};
    
    if(sum(both)>0)
        handles = [handles,x_and_y(1)];
        legends = {legends{:},'x \& y'};
    end
    
    if(sum(missing(:,1))>0)
        handles = [handles,missing_o(1)];
        legends = {legends{:},['Missing ',labels(o)]};
    end
    
    if(sum(missing(:,2))>0)
        handles = [handles, missing_u(1)];
        legends = {legends{:},['Missing ',labels(u)]};
    end

    
    legend(handles,legends,'interpreter','latex');
    xlabel(labels(o));
    ylabel(labels(u));
    
    title([labels(u),' given ', labels(o)]);
    axis  tight
    
    if(o==2)
        view([90 -90])
    end
end

