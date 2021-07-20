function [mu_all,r_all,rh_all,sh_all]=keepTrainingSetOnly(trial_ids,mu_all,r_all,rh_all,sh_all)

mu_all = mu_all(trial_ids,:);
r_all = r_all(trial_ids);
rh_all = rh_all(trial_ids);
sh_all = sh_all(trial_ids);