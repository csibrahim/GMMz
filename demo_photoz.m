
%%%%%%%%%%%%%% CONFIGURATION %%%%%%%%%%%%%%%%
rng(1);                                 % fix random seed
 
addpath VB_GMM/                         % path to the VB library
 
K = 100;                                % number of mixtures to use [required]
 

maxIter = 100;                          % maximum number of iterations
tol = 1e-10;                            % early stoping criteria if no significant improvement is gained
init= 'conditional';                    % how to initialize ('rand', 'conditional', 'kmeans')
cov = 'full';                           % type of covariance ('full', 'diag')
display=1;                              % display progress
 
 
trainSplit = 0.6;                       % percentage of data to use for training
validSplit = 0.2;                       % percentage of data to use for validation
testSplit  = 0.2;                       % percentage of data to use for testing

 
bins = 100;                            % number of samples to plot the CDFs and PDFs

%%%%%%%%%%%%%% SETUP %%%%%%%%%%%%%% 
 
dataPath = '../data/sdss_sample.csv';  % path to the data set, has to be in the following format m_1,m_2,..,m_k,e_1,e_2,...,e_k,z_spec
                                       % where m_i is the i-th magnitude, e_i is its associated uncertainty and z_spec is the spectroscopic redshift
                                       % [required]
                                        
% read data from file
X = csvread(dataPath);

[n,d] = size(X);

filters = (d-1)/2;

% transform the errors via a log function
X(:,filters+1:2*filters) = log(X(:,filters+1:2*filters));

% select training, validation and testing sets from the data
[training,validation,testing] = sample(n,trainSplit,validSplit,testSplit); 

%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%

% set training options
options.cyc=maxIter;
options.tol=tol;
options.init=init;
options.cov=cov;
options.display=display;

% fit the training data using K mixtures
model = gmmvar_missing(X(training,:),K, options);
%%%%%%%%%%%%%% COMPUTE METRICS %%%%%%%%%%%%%%

% use the model to to predict the d-th variable (i.e., redshift) using the
% first d-1 variables (i.e., the photometry)
[mu, sigma, prob_mu, mode, prob_mode, median, prob_median, values, PDF, CDF] = predict_VB(X(testing,:),1:d-1,d,model,bins);


Zspec = X(testing,end);

%root mean squared error, i.e. sqrt(mean(errors^2))
rmse = sqrt(metrics(Zspec,mu,prob_mu,@(y,mu,prob) (y-mu).^2)); 
 
% mean log likelihood 
mll = metrics(Zspec,mu,prob_mu,@(y,mu,prob) log(prob));
 
% fraction of data where |z_spec-z_phot|/(1+z_spec)<0.15
fr15 = metrics(Zspec,mu,prob_mu,@(y,mu,prob) 100*(abs(y-mu)./(y+1)<0.15));
 
% fraction of data where |z_spec-z_phot|/(1+z_spec)<0.05
fr05 = metrics(Zspec,mu,prob_mu,@(y,mu,prob) 100*(abs(y-mu)./(y+1)<0.05));
 
% bias, i.e. mean(errors)
bias = metrics(Zspec,mu,prob_mu,@(y,mu,prob) y-mu);
 
% print metrics for the entire data
fprintf('RMSE\t\tMLL\t\tFR15\t\tFR05\t\tBIAS\n')
fprintf('%f\t%f\t%f\t%f\t%f\n',rmse(end),mll(end),fr15(end),fr05(end),bias(end))
 
%%%%%%%%%%%%%% PLOTING %%%%%%%%%%%%%%%% 

% getting the limits to unify the plots
min_y = min([min(mu) min(mode) min(median)]);
max_y = min([max(mu) max(mode) max(median)]);

limits = [min(Zspec) max(Zspec) min_y max_y];

%%%%%%%%% the mean scatter plot %%%%%%%%%

% probability
figure('NumberTitle', 'off', 'Name', 'Mean');
subplot(1,2,1)
heat(Zspec,mu,prob_mu,0.01,1);
axis(limits);
title('Probability');xlabel('Spectroscopic Redshift');ylabel('Photometric Redshift');colormap jet;

% density
subplot(1,2,2)
heat(Zspec,mu,[],0.01,1);
axis(limits);
title('Density');xlabel('Spectroscopic Redshift');ylabel('Photometric Redshift');colormap jet;

%%%%%%%%% the median scatter plot %%%%%%%%%
% probability
figure('NumberTitle', 'off', 'Name', 'Median');
subplot(1,2,1)
heat(Zspec,median,prob_median,0.01,1);
axis(limits);
title('Probability');xlabel('Spectroscopic Redshift');ylabel('Photometric Redshift');colormap jet;

% density
subplot(1,2,2)
heat(Zspec,median,[],0.01,1);
axis(limits);
title('Density');xlabel('Spectroscopic Redshift');ylabel('Photometric Redshift');colormap jet;

%%%%%%%%% the mode scatter plot %%%%%%%%%
% probability
figure('NumberTitle', 'off', 'Name', 'Mode');
subplot(1,2,1)
heat(Zspec,mode,prob_mode,0.01,1);
axis(limits);
title('Probability');xlabel('Spectroscopic Redshift');ylabel('Photometric Redshift');colormap jet;

% density
subplot(1,2,2)
heat(Zspec,mode,[],0.01,1);
axis(limits);
title('Density');xlabel('Spectroscopic Redshift');ylabel('Photometric Redshift');colormap jet;


%%%%%%%%% CDF and PDF %%%%%%%%%

% getting the prediction where the mean has the minimum probability
[~,order] = sort(prob_mu);
id = order(1);

% the single Gaussian pdf
singlePDF = mvnpdf(values(id,:)',mu(id),sigma(id))';

% using two y-axes and setting their colors to blak [0 0 0 ]
fig = figure;
set(fig,'defaultAxesColorOrder',[[0 0 0]; [0 0 0]]);

hold on;

% plot the CDF against the right y-axis
yyaxis right
cdf_plot = plot(values(id,:),CDF(id,:),'-','Color',[255 165 0]/255,'LineWidth',2);
ylabel('CDF')

% plot the PDF against the left y-axis
yyaxis left
pdf_plot = plot(values(id,:),PDF(id,:),'r-','LineWidth',2); % the full PDF
norm_plot = plot(values(id,:),singlePDF,'k-','LineWidth',2);% the single PDF
ylabel('PDF')

% the value of the mean and it's probability
mu_marker = plot(mu(id),0,'kx','MarkerSize',10); % mark the value
p_mu_marker = plot(mu(id),prob_mu(id),'ko','MarkerSize',10,'MarkerFaceColor','k'); % mark the probability
plot([mu(id);mu(id)],[0;prob_mu(id)],'k:','LineWidth',2); % mark a vertical line

% the value of the mode and it's probability
mode_marker = plot(mode(id),0,'rx','MarkerSize',10); % mark the value
p_mode_marker = plot(mode(id),prob_mode(id),'ko','MarkerSize',10,'MarkerFaceColor','r'); % mark the probability
plot([mode(id);mode(id)],[0;max(PDF(id,:))],'r:','LineWidth',2); % mark a vertical line

% the value of the median and it's probability
median_marker = plot(median(id),0,'gx','MarkerSize',10); % mark the value
p_median_marker = plot(median(id),prob_median(id),'ko','MarkerSize',10,'MarkerFaceColor','g'); % mark the probability
plot([median(id);median(id)],[0;prob_median(id)],'g:','LineWidth',2); % mark a vertical line

% the actual value
trueZ_marker = plot(Zspec(id),0,'bx','MarkerSize',10); % mark the value
plot([Zspec(id);Zspec(id)],[0;max(PDF(id,:))],'b-','LineWidth',2); % mark a vertical line

axis tight;

handles = [cdf_plot, pdf_plot, norm_plot, p_mu_marker, p_mode_marker, p_median_marker, mu_marker, mode_marker, median_marker, trueZ_marker];
legends = {'CDF', 'Full PDF','Single PDF','p(Mean)','p(Mode)','p(Median)','Mean','Mode','Median','True $z_{s}$'};
legend(handles,legends,'interpreter','latex','Location','northwest');


%%%%%%%%% the qq plot %%%%%%%%%

% theoratical values
percentiles = 0.01:0.01:0.99;
z = percentile(X(testing,:),1:d-1,d,model,percentiles);

% emprical values
qq_d = mean(repmat(Zspec,1,size(z,2))<z);

figure;
plot([0 percentiles 1],[0 percentiles 1],'k--',[0 percentiles 1],[0 qq_d 1],'.-'); axis tight

%%%%%%%%% metrics as functions of data percentage %%%%%%%%%

% for every 5% percent of the data
x = [1 5:5:100];
ind = round(x*length(rmse)/100);
 
figure;plot(x,rmse(ind),'o-');xlabel('Percentage of Data');ylabel('RMSE');
figure;plot(x,mll(ind),'o-');xlabel('Percentage of Data');ylabel('MLL');
figure;plot(x,fr05(ind),'o-');xlabel('Percentage of Data');ylabel('FR05');
figure;plot(x,fr15(ind),'o-');xlabel('Percentage of Data');ylabel('FR15');
figure;plot(x,bias(ind),'o-');xlabel('Percentage of Data');ylabel('BIAS');