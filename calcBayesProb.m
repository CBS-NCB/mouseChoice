function prob_out=calcBayesProb(mu,r,s,params,mode,bessel_table,bessel_coords)
% compute model probability for a full model; requires Parallel Processing
% Toolbox

%unpack, usual number of parameters - 8.
hs = params(1);% stimulus history coefficient (h_s)
hr = params(2);% choice history coefficient (h_r)
sgmax = params(3);% concentration for x (kappa_R)
sgmay = params(4);% concentration for y (kappa_L)
sgmapr = params(5);% concentration of history prior (kappa_h)
if length(params)==5 % for test purposes only, not used in the manuscript
    x0 = 0; y0 = 0;
elseif length(params)==7 % for test purposes only, not used in the manuscript
    x0 = params(6);% perceptual bias for x (b_R)
    y0 = params(7);% perceptual bias for y (b_L)
elseif length(params)==8
    x0 = params(6);% perceptual bias for x (b_R)
    y0 = params(7);% perceptual bias for y (b_L)
    sgmab = params(8);% parameter of bias distribution (kappa_b)
end

% if computing on CPU, use anonymous functions (1.convenience, 2.precision of besseli)
% also, equations in the "CPU" section are more readable, and mirrors exactly the
% operations in "GPU" section
if strcmp(mode,'CPU'),
    
    h = hs*s+hr*r;% the only way history terms appear in calculations
    
    % unique conditions:
    all_contions = [mu h];
    [unq_cond,~,ind_orig_in_unique] = unique(all_contions,'rows');
    
    mu1 = unq_cond(:,1);
    mu2 = unq_cond(:,2);
    hist = unq_cond(:,3);
    
    nCond = size(unq_cond,1); 
    prob = zeros(length(unq_cond),1);%choose-1 probability, allocate array

    % function definitions (these don't change with loop iteration):
    % von Mises distribution
    Cx = 1/(2*pi*besseli(0,sgmax));
    Cy = 1/(2*pi*besseli(0,sgmay));
    Cb = 1/(2*pi*besseli(0,sgmab));
    px = @(x,mu) Cx * exp((sgmax)*cos(x-mu));
    py = @(x,mu) Cy * exp((sgmay)*cos(x-mu));
    pb = @(x,y) Cb^2 * exp(sgmab*(cos(x)-cos(y)));
    
    parfor iCond = 1:nCond
        
        this_mu1 = mu1(iCond);
        this_mu2 = mu2(iCond);
        this_hist = hist(iCond);
        
        
        Cprior = 1/(2*pi*besseli(0,sgmapr*this_hist));
        prior = @(x,y) Cprior^2*exp(sgmapr*this_hist*(cos(x)-cos(y)));
        
        % numerator function -- note the order of (y,x); y has to come first
        % for integral2
        pnumf = @(y,x) px(x-x0,this_mu1).*py(y-y0,this_mu2).*prior(x,y).*pb(x,y);
        
        % integrals
        xmax = @(y) y;
        xmin = @(y) -y;
        
        abstol = 1e-4;
        reltol = 1e-3;
        pnum(iCond) = integral2(pnumf,0,pi,xmin,xmax,'AbsTol',abstol,'RelTol',reltol)+integral2(pnumf,-pi,0,xmax,xmin,'AbsTol',abstol,'RelTol',reltol);%a,b,c,d,u,v
        normc(iCond) =  integral2(pnumf,-pi,pi,-pi,pi,'AbsTol',abstol,'RelTol',reltol);
        
        % strictly speaking normalization is not necessary and is taken care of by the constants; more of a sanity check 
        prob(iCond) = pnum(iCond)/normc(iCond);
        
    end
    
    
else% GPU: 
    % prepare arrays for integration, use precalculated bessel function values
    
    h = hs*s+hr*r;% the way history terms appear in calculations
    
    % unique conditions:
    all_contions = [mu h];
    [unq_cond,~,ind_orig_in_unique] = unique(all_contions,'rows');
    mu1 = unq_cond(:,1);
    mu2 = unq_cond(:,2);
    hist = unq_cond(:,3);
    nCond = size(unq_cond,1);
    
    prob = zeros(length(unq_cond),1,'gpuArray');%choose-1 probability, allocate array

    % -I. Bessel function values
    bessel_table = gpuArray(bessel_table);
    [~,idx_x] = min(abs(bessel_table-sgmax));
    [~,idx_y] = min(abs(bessel_table-sgmay));
    [~,idx_b] = min(abs(bessel_coords-sgmab));
    besx = bessel_table(idx_x);
    besy = bessel_table(idx_y);
    besb = bessel_table(idx_b);
    bespr = zeros(1,1,nCond,'gpuArray');
    bessel_coords = gpuArray(bessel_coords);
    [~,idx_pr]=min(abs(bsxfun(@minus,bessel_coords,(sgmapr*hist)')));
    bespr(1,1,:) = bessel_table(idx_pr); clear idx_pr
    
    % sequence of operations is such that they take up less GPU memory (which
    % is unexpectedly an issue)
    
    % 0. create X3 and Y3 arrays, used in all the calculations below
    N=300;
    xvals = gpuArray.linspace(-pi, pi, N);
    yvals = gpuArray.linspace(-pi, pi, N);
    [X, Y] = meshgrid(xvals, yvals);
    xspacing = 2*pi/N;
    yspacing = 2*pi/N;
    
    % I. compute the mu1-containing term
    this_mu1(1,1,:) = gpuArray(mu1+x0); 
    
    % II. compute the mu2-containing term
    this_mu2(1,1,:) = gpuArray(mu2+y0); 
    
    % III. keep only their sum
    tmp12_1 = sgmax*cos(bsxfun(@minus,X,this_mu1));
    tmp12_2 = sgmay*cos(bsxfun(@minus,Y,this_mu2));
    tmp12 = tmp12_1 + tmp12_2; clear tmp12_1 tmp12_2

    % IV. calculate the history term coefficient
    this_hist(1,1,:) = gpuArray(hist);
    
    % V. calculate the history term
    tmp3 = bsxfun(@times,cos(X)-cos(Y),this_hist*sgmapr) + tmp12 + sgmab*(cos(X)-cos(Y)); clear tmp12 % note, must support broadcasting

    % VI. calculate a coefficient for the full expression
    C1C2C3 = 1./(2*pi*besx * 2*pi*besy * (2*pi*bespr).^2 * (2*pi*besb)^2);
    
    % VII. calculate the full expression
    tmp4 = bsxfun(@times,exp(tmp3),C1C2C3); clear tmp3

    
    % mask for half-integral
    mask = gpuArray(abs(X)<=abs(Y));

    % integral over the full space
    Z1 = trapz(tmp4)*yspacing;
    int_full = trapz(Z1)*xspacing; clear Z1
    
    % integral over the half space (|X|<|Y|)
    func_half = bsxfun(@mtimes,tmp4,mask); clear tmp4 mask
    Z1_half = trapz(func_half)*yspacing; clear func_half
    int_half = trapz(Z1_half)*xspacing; clear Z1_half
    
    prob = squeeze(int_half./int_full);
    
end

prob_out = gather(prob(ind_orig_in_unique));