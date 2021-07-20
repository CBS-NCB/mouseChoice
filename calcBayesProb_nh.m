function prob_out=calcBayesProb_nh(mu,params,mode,bessel_table,bessel_coords)
% model probability with no history biases; requires Parallel Computing Toolbox

% unpack parameters:
sgmax = params(1);% concentration for x (aka kappa_R)
sgmay = params(2);% concentration for y (aka kappa_L) 
x0 = params(3);% perceptual bias for x (aka b_R)
y0 = params(4);% perceptual bias for y (aka b_L)
sgmab = params(5);% parameter for bias distribution

% if computing on CPU, we use anonymous functions (1.convenience, 2.precision of besseli, 
% 3. CPU regime is not intended for fitting, but for simulation, hence there is only one iteration
% over the dataset, and the time of calculation is not an issue)
% Also, equations in "CPU" section are more readable, and mirrors exactly the
% operations in "GPU" section
if strcmp(mode,'CPU'),

    % unique conditions:
    all_contions = mu;
    [unq_cond,~,ind_orig_in_unique] = unique(all_contions,'rows');
    
    mu1 = unq_cond(:,1);
    mu2 = unq_cond(:,2);
    
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
        
        % numerator function -- note the order of (y,x); y has to come first
        % for integral2
        pnumf = @(y,x) px(x-x0,this_mu1).*py(y-y0,this_mu2).*pb(x,y);
        
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
    
    % unique conditions:
    all_contions = mu;
    [unq_cond,~,ind_orig_in_unique] = unique(all_contions,'rows');
    mu1 = unq_cond(:,1);
    mu2 = unq_cond(:,2);
    nCond = size(unq_cond,1);
    
    prob = zeros(length(unq_cond),1,'gpuArray');%choose-1 probability, allocate array

    % -I. Bessel function values
    bessel_table = gpuArray(bessel_table);
    bessel_coords = gpuArray(bessel_coords);
    [~,idx_x] = min(abs(bessel_coords-sgmax));
    [~,idx_y] = min(abs(bessel_coords-sgmay));
    [~,idx_b] = min(abs(bessel_coords-sgmab));
    besx = bessel_table(idx_x);
    besy = bessel_table(idx_y);
    besb = bessel_table(idx_b);
    
    % sequence of the following operations is such that they take less GPU memory 
    % (which is unexpectedly an issue)
    
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
    
    % III. add the bias term and keep only the sum
    tmp12 = sgmax*cos(bsxfun(@minus,X,this_mu1))+sgmay*cos(bsxfun(@minus,Y,this_mu2)) + sgmab*(cos(X)-cos(Y));  % note, must support broadcasting
    clear this_mu1 this_mu2
   
    % IV. calculate a coefficient for the full expression
    C1C2C3 = 1./(2*pi*besx * 2*pi*besy * (2*pi*besb)^2 * ones(1,1,nCond,'gpuArray') );
    
    % V. calculate the full expression
    tmp3 = bsxfun(@times,exp(tmp12),C1C2C3);
    
    % compute mask for integration
    mask = gpuArray(abs(X)<=abs(Y));% half-space over which to integrate

    % integral over the full space
    Z1 = trapz(tmp3)*yspacing;
    int_full = trapz(Z1)*xspacing; clear Z1
    
    % integral over the half space (|X|<|Y|)
    func_half = bsxfun(@mtimes,tmp3,mask); clear tmp3 mask
    Z1_half = trapz(func_half)*yspacing; clear func_half
    int_half = trapz(Z1_half)*xspacing; clear Z1_half
    
    
    prob = squeeze(int_half./int_full);
    
end


prob_out = gather(prob(ind_orig_in_unique));