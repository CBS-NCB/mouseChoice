function [mu_all,rh_all,sh_all,r_all] = occurrence_check(mu_all,rh_all,sh_all,r_all,Nmax)
% if the number of unique conditions is too large, remove conditions with
% too few trials

if nargin<5,
    Nmax = 5000;
end


h = [sh_all rh_all];
all_contions = [mu_all h];
[unq_cond,ind_unique_in_orig,ind_orig_in_unique] = unique(all_contions,'rows');

if length(ind_orig_in_unique)>Nmax,
    
    disp(['number of unique conditions larger than ' num2str(Nmax) ': ignoring those with fewer than 3 trials']);

    for i=1:length(ind_orig_in_unique)
        count(i) = sum(ind_orig_in_unique==ind_orig_in_unique(i));
    end

    unq_ids_to_exclude = ind_orig_in_unique(count<3);
    
    % how many actually?
    unq_unq_ids_to_exclude = unique(unq_ids_to_exclude);
    
    exclude_conditions = unq_cond(unq_unq_ids_to_exclude,:);
    
    exclude_ids = ismember(all_contions,exclude_conditions,'rows');
    
    mu_all(exclude_ids,:) = [];
    sh_all(exclude_ids) = [];
    rh_all(exclude_ids) = [];
    r_all(exclude_ids) = [];
end

