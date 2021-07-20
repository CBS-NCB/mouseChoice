%% demo that fits a model of choice in a 2AFC orientation discrimination task (in a 2d stim space of angles)

% model input: two-dimensional cyclical variable (orientations of two targets, left and right)
% model output: binary choice (left choice = 0, right choice = 1)
% ideally, the subject must choose the target whose angle is closer to 0,
% i.e. choice = heaviside(|angle_L|-|angle_R|)
% the model accounts for possible perceptual biases, different for each target,
% for the unequal certainty in the percepts, and for the influence of
% previous choices and rewards

%% what we do here:
% 1. fit two models to a dataset: a no-history and a history-included model
% (once each), main goal of this script is to show how to obtain parameter
% vectors ('theta_hat','theta_hat_nh') that fully describe the respective
% model
% 2. sample decisions in same exact conditions as the data
% 3. plot p(r) from the data against p(r) from both simulations

% 2-3 are only to get the rough idea about the fit; deeper analysis/
% manuscript plots are generated elsewhere

% other things not covered in this script: (1) RT-dependent models, (2)
% cross-validated models, 
%% 
animal = 15312;
training_set_only = false;
%%
addpath('data/');
load(['B' num2str(animal) '.mat']); % raw data
load('sim_benchmark_data.mat','bessel_coords','bessel_table'); % grid of constants needed to compute integrals in calcBayesProb*
%%
mu_all = B.ori_clean;
r_all = B.choicedir_clean;
rh_all = B.rh;
sh_all = B.sh;
if ismember(animal,[15301;15352;16032;16001;16199;17007;17010;17037;17149;18099;18100;18101])% horizontal-target mice
    mu_all = fliplr(mu_all);
    sh_all = -sh_all;
end
%%
if training_set_only,
    [mu_all,r_all,rh_all,sh_all]=keepTrainingSetOnly(B.train_id>0,mu_all,r_all,rh_all,sh_all);
end
%% show p(r=1) surface as is in the data
[~,~,~,p_data_surf]=showProbRightSurface(mu_all,r_all,10);
%% 1. no-history model: fit

lb =[ 0; 0; -pi; -pi; -10];
ub =[ 10; 10; pi; pi; 10];
options = optimoptions('fmincon','Display', 'iter','TolFun',1e-5,'TolX',1e-5,'MaxFunEval',500,'MaxIter',100);

tic
for i = 1:length(lb),
    these_params0(i) = ( (rand-0.5)*(ub(i)-lb(i)) + (lb(i)+ub(i))/2 )/10;
end
these_params0 = reshape(these_params0,[],1);
    
clear prob crossEnt
prob = @(params) calcBayesProb_nh(mu_all,params,'GPU',bessel_table,bessel_coords);
crossEnt = @(params) -(r_all'*log(prob(params)) + (1-r_all)'*log(1-prob(params)));
    
[theta_hat_nh, fval0]=fmincon(crossEnt, these_params0, [],[],[],[],lb,ub,[],options);
total_time_nh = toc
%% no-history model: show fit results
p_nh_all = calcBayesProb_nh(mu_all,theta_hat_nh,'GPU',bessel_table,bessel_coords);
r_nh_sim = binornd(1,p_nh_all);
[~,~,~,p_nh_surf]=showProbRightSurface(mu_all,r_nh_sim,10);
%% 2. model with history: fit

lb = [-2;-2; 0; 0; 0; -pi ; -pi; -10];
ub = [2; 2; 10; 10; 10; pi; pi; 10];
options = optimoptions('fmincon','Display', 'iter','TolFun',1e-4,'TolX',1e-4,'MaxFunEval',500,'MaxIter',100);

% check once if there are not too few trials with a given condition
[mu_all,rh_all,sh_all,r_all] = occurrence_check(mu_all,rh_all,sh_all,r_all);

for i = 1:length(lb),
    these_params0(i) = ( (rand-0.5)*(ub(i)-lb(i)) + (lb(i)+ub(i))/2 )/10;
end
these_params0 = reshape(these_params0,[],1);
    
clear prob crossEnt
prob = @(params) calcBayesProb(mu_all,rh_all,sh_all,params,'GPU',bessel_table,bessel_coords);
crossEnt = @(params) -(r_all'*log(prob(params)) + (1-r_all)'*log(1-prob(params)));

[theta_hat, fval0]=fmincon(crossEnt, these_params0, [],[],[],[],lb,ub,[],options);
total_time = toc
%% model with history: show fit results
p_all = calcBayesProb(mu_all,rh_all,sh_all,theta_hat,'GPU',bessel_table,bessel_coords);
r_sim = binornd(1,p_all);
[~,~,~,p_surf]=showProbRightSurface(mu_all,r_sim,10);
%%
figure,
subplot(1,3,1); 
imagesc(p_data_surf); caxis([0 1]); title('p(r), data')
subplot(1,3,2)
imagesc(p_nh_surf); caxis([0 1]); title('p(r), no hist model')
subplot(1,3,3)
imagesc(p_surf); caxis([0 1]); title('p(r), hist model')
%%
fn = ['models/m' num2str(animal) '_fitted.mat'];
save(fn,'theta_hat','theta_hat_nh','p_surf','p_nh_surf','p_data_surf','r_sim','r_nh_sim','mu_all','rh_all','sh_all','r_all','p_all','p_nh_all','total_time','total_time_nh');