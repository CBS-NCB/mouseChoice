function [p0,unqpairs,nPair,p_surf]=showProbRightSurface(mu_all,r_all,n_min)

% show the surface of performance as in the data, this is for comparison
% with a similar surface obtained theoretically

% n_min sets minimum number of trials with a specific pair of angles
% (mu_all), if the number of trials with such condition is less than n_min,
% the condition is discarded

unqpairs = unique(mu_all,'rows');

for iPair = 1:size(unqpairs,1)
    ind = ismember(mu_all,unqpairs(iPair,:),'rows');
    nPair(iPair) = sum(ind);
    p(iPair) = sum(r_all(ind)==1)/nPair(iPair);
end

% put p on a grid of (x,y)
[x,y] = meshgrid(unique(mu_all(:,1)),unique(mu_all(:,2)));
p0 = nan(size(x));
for iPair = 1:size(unqpairs,1)
    if nPair(iPair)<n_min
        continue
    end
    ind = ismember([x(:) y(:)],unqpairs(iPair,:),'rows');
    p0(ind) = p(iPair);
end

x_mesh = x(~isnan(p0));
y_mesh = y(~isnan(p0));

% x_mesh and y_mesh are missing some nodes, interpolate 
xu = unique(x_mesh);
yu = unique(y_mesh);
disp(sum(~isnan(p0(:))))
[x,y] = meshgrid(xu,yu);
p_surf=griddata(x_mesh,y_mesh,p0(~isnan(p0)),x,y,'cubic');

figure, 
surf(x,y,p_surf);
xlabel('right angle x2, rad')
ylabel('left angle x2, rad')
zlabel('prob. right choice')