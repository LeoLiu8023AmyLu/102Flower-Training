function F = hogcalculator(img, cellpw, cellph, nblockw, nblockh,...
    nthet, overlap, isglobalinterpolate, issigned, normmethod)
% HOGCALCULATOR calculate R-HOG feature vector of an input image using the
% procedure presented in Dalal and Triggs's paper in CVPR 2005.
%

% Author:   timeHandle
% Time:     March 24, 2010
%           May 12£¬2010 update.
%
%       this copy of code is written for my personal interest, which is an
%       original and inornate realization of [Dalal CVPR2005]'s algorithm
%       without any optimization. I just want to check whether I understand
%       the algorithm really or not, and also do some practices for knowing
%       matlab programming more well because I could be called as 'novice'.
%       OpenCV 2.0 has realized Dalal's HOG algorithm which runs faster
%       than mine without any doubt, ¨r(¨s¨Œ¨t)¨q . Ronan pointed a error in
%       the code£¬thanks for his correction. Note that at the end of this
%       code, there are some demonstration code£¬please remove in your work.


%
% F = hogcalculator(img, cellpw, cellph, nblockw, nblockh,
%    nthet, overlap, isglobalinterpolate, issigned, normmethod)
%
% IMG:
%       IMG is the input image.
%
% CELLPW, CELLPH:
%       CELLPW and CELLPH are cell's pixel width and height respectively.
%
% NBLOCKW, NBLCOKH:
%       NBLOCKW and NBLCOKH are block size counted by cells number in x and
%       y directions respectively.
%
% NTHET, ISSIGNED:
%       NTHET is the number of the bins of the histogram of oriented
%       gradient. The histogram of oriented gradient ranges from 0 to pi in
%       'unsigned' condition while to 2*pi in 'signed' condition, which can
%       be specified through setting the value of the variable ISSIGNED by
%       the string 'unsigned' or 'signed'.
%
% OVERLAP:
%       OVERLAP is the overlap proportion of two neighboring block.
%
% ISGLOBALINTERPOLATE:
%       ISGLOBALINTERPOLATE specifies whether the trilinear interpolation
%       is done in a single global 3d histogram of the whole detecting
%       window by the string 'globalinterpolate' or in each local 3d
%       histogram corresponding to respective blocks by the string
%       'localinterpolate' which is in strict accordance with the procedure
%       proposed in Dalal's paper. Interpolating in the whole detecting
%       window requires the block's sliding step to be an integral multiple
%       of cell's width and height because the histogram is fixing before
%       interpolate. In fact here the so called 'global interpolation' is
%       a notation given by myself. at first the spatial interpolation is
%       done without any relevant to block's slide position, but when I was
%       doing calculation while OVERLAP is 0.75, something occurred and
%       confused me o__O"¡­ . This let me find that the operation I firstly
%       did is different from which mentioned in Dalal's paper. But this
%       does not mean it is incorrect ^¡ò^, so I reserve this. As for name,
%       besides 'global interpolate', any others would be all ok, like 'Lady GaGa'
%       or what else, :-).
%
% NORMMETHOD£º
%       NORMMETHOD is the block histogram normalized method which can be
%       set as one of the following strings:
%               'none', which means non-normalization;
%               'l1', which means L1-norm normalization;
%               'l2', which means L2-norm normalization;
%               'l1sqrt', which means L1-sqrt-norm normalization;
%               'l2hys', which means L2-hys-norm normalization.
% F£º
%       F is a row vector storing the final histogram of all of the blocks
%       one by one in a top-left to bottom-right image scan manner, the
%       cells histogram are stored in the same manner in each block's
%       section of F.
%
% Note that CELLPW*NBLOCKW and CELLPH*NBLOCKH should be equal to IMG's
% width and height respectively.
%
% Here is a demonstration, which all of parameters are set as the
% best value mentioned in Dalal's paper when the window detected is 128*64
% size(128 rows, 64 columns):
%       F = hogcalculator(window, 8, 8, 2, 2, 9, 0.5,
%                               'localinterpolate', 'unsigned', 'l2hys');
% Also the function can be called like:
%       F = hogcalculator(window);
% the other parameters are all set by using the above-mentioned "dalal's
% best value" as default.
%

if nargin < 2
    % set default parameters value.
    cellpw = 8;
    cellph = 8;
    nblockw = 2;
    nblockh = 2;
    nthet = 9;
    overlap = 0.5;
    isglobalinterpolate = 'localinterpolate';
    issigned = 'unsigned';
    normmethod = 'l2hys';
else
    if nargin < 10
        error('Input parameters are not enough.');
    end
end

% check parameters's validity.
[M, N, K] = size(img);
if mod(M,cellph*nblockh) ~= 0
    error('IMG''s height should be an integral multiple of CELLPH*NBLOCKH.');
end
if mod(N,cellpw*nblockw) ~= 0
    error('IMG''s width should be an integral multiple of CELLPW*NBLOCKW.');
end
if mod((1-overlap)*cellpw*nblockw, cellpw) ~= 0 ||...
        mod((1-overlap)*cellph*nblockh, cellph) ~= 0
    str1 = 'Incorrect OVERLAP or ISGLOBALINTERPOLATE parameter';
    str2 = ', slide step should be an intergral multiple of cell size';
    error([str1, str2]);
end

% set the standard deviation of gaussian spatial weight window.
delta = cellpw*nblockw * 0.5;

% calculate gradient scale matrix.
hx = [-1,0,1];
hy = -hx';
gradscalx = imfilter(double(img),hx);
gradscaly = imfilter(double(img),hy);

%
if K > 1
        maxgrad = sqrt(double(gradscalx.*gradscalx + gradscaly.*gradscaly));
        [gradscal, gidx] = max(maxgrad,[],3);
        gxtemp = zeros(M,N);
        gytemp = gxtemp;
        for kn = 1:K
            ttempx = gradscalx(:,:,kn);
            ttempy = gradscaly(:,:,kn);
            tmpindex = find(gidx==kn);
            gxtemp(tmpindex) = ttempx(tmpindex);
            gytemp(tmpindex) =ttempy(tmpindex);
        end
        gradscalx = gxtemp;
        gradscaly = gytemp;
else
    gradscal = sqrt(double(gradscalx.*gradscalx + gradscaly.*gradscaly));
end

 

% calculate gradient orientation matrix.
% plus small number for avoiding dividing zero.
gradscalxplus = gradscalx+ones(size(gradscalx))*0.0001;
gradorient = zeros(M,N);
% unsigned situation: orientation region is 0 to pi.
if strcmp(issigned, 'unsigned') == 1
    gradorient =...
        atan(gradscaly./gradscalxplus);
    gradorient(gradorient<0) = gradorient(gradorient<0)+pi;
    or = 1;
else
    % signed situation: orientation region is 0 to 2*pi.
    if strcmp(issigned, 'signed') == 1
        idx = find(gradscalx >= 0 & gradscaly >= 0);
        gradorient(idx) = atan(gradscaly(idx)./gradscalxplus(idx));
        idx = find(gradscalx < 0);
        gradorient(idx) = atan(gradscaly(idx)./gradscalxplus(idx)) + pi;
        idx = find(gradscalx >= 0 & gradscaly < 0);
        gradorient(idx) = atan(gradscaly(idx)./gradscalxplus(idx)) + 2*pi;
        or = 2;
    else
        error('Incorrect ISSIGNED parameter.');
    end
end

% calculate block slide step.
xbstride = cellpw*nblockw*(1-overlap);
ybstride = cellph*nblockh*(1-overlap);
xbstridend = N - cellpw*nblockw + 1;
ybstridend = M - cellph*nblockh + 1;

% calculate the total blocks number in the window detected, which is
% ntotalbh*ntotalbw.
ntotalbh = ((M-cellph*nblockh)/ybstride)+1;
ntotalbw = ((N-cellpw*nblockw)/xbstride)+1;

% generate the matrix hist3dbig for storing the 3-dimensions histogram. the
% matrix covers the whole image in the 'globalinterpolate' condition or
% covers the local block in the 'localinterpolate' condition. The matrix is
% bigger than the area where it covers by adding additional elements
% (corresponding to the cells) to the surround for calculation convenience.
if strcmp(isglobalinterpolate, 'globalinterpolate') == 1
    ncellx = N / cellpw;
    ncelly = M / cellph;
    hist3dbig = zeros(ncelly+2, ncellx+2, nthet+2);
    F = zeros(1, (M/cellph-1)*(N/cellpw-1)*nblockw*nblockh*nthet);
    glbalinter = 1;
else
    if strcmp(isglobalinterpolate, 'localinterpolate') == 1
        hist3dbig = zeros(nblockh+2, nblockw+2, nthet+2);
        F = zeros(1, ntotalbh*ntotalbw*nblockw*nblockh*nthet);
        glbalinter = 0;
    else
        error('Incorrect ISGLOBALINTERPOLATE parameter.')
    end
end

% generate the matrix for storing histogram of one block;
sF = zeros(1, nblockw*nblockh*nthet);

% generate gaussian spatial weight.
[gaussx, gaussy] = meshgrid(0:(cellpw*nblockw-1), 0:(cellph*nblockh-1));
weight = exp(-((gaussx-(cellpw*nblockw-1)/2)...
    .*(gaussx-(cellpw*nblockw-1)/2)+(gaussy-(cellph*nblockh-1)/2)...
    .*(gaussy-(cellph*nblockh-1)/2))/(delta*delta));

% vote for histogram. there are two situations according to the interpolate
% condition('global' interpolate or local interpolate). The hist3d which is
% generated from the 'bigger' matrix hist3dbig is the final histogram.
if glbalinter == 1
    xbstep = nblockw*cellpw;
    ybstep = nblockh*cellph;
else
    xbstep = xbstride;
    ybstep = ybstride;
end
% block slide loop
for btly = 1:ybstep:ybstridend
    for btlx = 1:xbstep:xbstridend
        for bi = 1:(cellph*nblockh)
            for bj = 1:(cellpw*nblockw)
               
                i = btly + bi - 1;
                j = btlx + bj - 1;
                gaussweight = weight(bi,bj);
               
                gs = gradscal(i,j);
                Go = gradorient(i,j);
               
                if glbalinter == 1
                    jorbj = j;
                    iorbi = i;
                else
                    jorbj = bj;
                    iorbi = bi;
                end
               
                % calculate bin index of hist3dbig
                binx1 = floor((jorbj-1+cellpw/2)/cellpw) + 1;
                biny1 = floor((iorbi-1+cellph/2)/cellph) + 1;
                binz1 = floor((go+(or*pi/nthet)/2)/(or*pi/nthet)) + 1;
               
                if gs < 1E-5
                    continue;
                end
               
                binx2 = binx1 + 1;
                biny2 = biny1 + 1;
                binz2 = binz1 + 1;
               
                x1 = (binx1-1.5)*cellpw + 0.5;
                y1 = (biny1-1.5)*cellph + 0.5;
                z1 = (binz1-1.5)*(or*pi/nthet);
               
                % trilinear interpolation.
                hist3dbig(biny1,binx1,binz1) =...
                    hist3dbig(biny1,binx1,binz1) + gs*gaussweight...
                    * (1-(jorbj-x1)/cellpw)*(1-(iorbi-y1)/cellph)...
                    *(1-(go-z1)/(or*pi/nthet));
                hist3dbig(biny1,binx1,binz2) =...
                    hist3dbig(biny1,binx1,binz2) + gs*gaussweight...
                    * (1-(jorbj-x1)/cellpw)*(1-(iorbi-y1)/cellph)...
                    *((go-z1)/(or*pi/nthet));
                hist3dbig(biny2,binx1,binz1) =...
                    hist3dbig(biny2,binx1,binz1) + gs*gaussweight...
                    * (1-(jorbj-x1)/cellpw)*((iorbi-y1)/cellph)...
                    *(1-(go-z1)/(or*pi/nthet));
                hist3dbig(biny2,binx1,binz2) =...
                    hist3dbig(biny2,binx1,binz2) + gs*gaussweight...
                    * (1-(jorbj-x1)/cellpw)*((iorbi-y1)/cellph)...
                    *((go-z1)/(or*pi/nthet));
                hist3dbig(biny1,binx2,binz1) =...
                    hist3dbig(biny1,binx2,binz1) + gs*gaussweight...
                    * ((jorbj-x1)/cellpw)*(1-(iorbi-y1)/cellph)...
                    *(1-(go-z1)/(or*pi/nthet));
                hist3dbig(biny1,binx2,binz2) =...
                    hist3dbig(biny1,binx2,binz2) + gs*gaussweight...
                    * ((jorbj-x1)/cellpw)*(1-(iorbi-y1)/cellph)...
                    *((go-z1)/(or*pi/nthet));
                hist3dbig(biny2,binx2,binz1) =...
                    hist3dbig(biny2,binx2,binz1) + gs*gaussweight...
                    * ((jorbj-x1)/cellpw)*((iorbi-y1)/cellph)...
                    *(1-(go-z1)/(or*pi/nthet));
                hist3dbig(biny2,binx2,binz2) =...
                    hist3dbig(biny2,binx2,binz2) + gs*gaussweight...
                    * ((jorbj-x1)/cellpw)*((iorbi-y1)/cellph)...
                    *((go-z1)/(or*pi/nthet));
            end
        end
       
        % In the local interpolate condition. F is generated in this block
        % slide loop. hist3dbig should be cleared in each loop.
        if glbalinter == 0
            if or == 2
                hist3dbig(:,:,2) = hist3dbig(:,:,2)...
                    + hist3dbig(:,:,nthet+2);
                hist3dbig(:,:,(nthet+1)) =...
                    hist3dbig(:,:,(nthet+1)) + hist3dbig(:,:,1);
            end
            hist3d = hist3dbig(2:(nblockh+1), 2:(nblockw+1), 2:(nthet+1));
            for ibin = 1:nblockh
                for jbin = 1:nblockw
                    idsF = nthet*((ibin-1)*nblockw+jbin-1)+1;
                    idsF = idsF:(idsF+nthet-1);
                    sF(idsF) = hist3d(ibin,jbin,:);
                end
            end
            iblock = ((btly-1)/ybstride)*ntotalbw +...
                ((btlx-1)/xbstride) + 1;
            idF = (iblock-1)*nblockw*nblockh*nthet+1;
            idF = idF:(idF+nblockw*nblockh*nthet-1);
            F(idF) = sF;
            hist3dbig(:,:,:) = 0;
        end
    end
end

% In the global interpolate condition. F is generated here outside the
% block slide loop
if glbalinter == 1
    ncellx = N / cellpw;
    ncelly = M / cellph;
    if or == 2
        hist3dbig(:,:,2) = hist3dbig(:,:,2) + hist3dbig(:,:,nthet+2);
        hist3dbig(:,:,(nthet+1)) = hist3dbig(:,:,(nthet+1)) + hist3dbig(:,:,1);
    end
    hist3d = hist3dbig(2:(ncelly+1), 2:(ncellx+1), 2:(nthet+1));
   
    iblock = 1;
    for btly = 1:ybstride:ybstridend
        for btlx = 1:xbstride:xbstridend
            binidx = floor((btlx-1)/cellpw)+1;
            binidy = floor((btly-1)/cellph)+1;
            idF = (iblock-1)*nblockw*nblockh*nthet+1;
            idF = idF:(idF+nblockw*nblockh*nthet-1);
            for ibin = 1:nblockh
                for jbin = 1:nblockw
                    idsF = nthet*((ibin-1)*nblockw+jbin-1)+1;
                    idsF = idsF:(idsF+nthet-1);
                    sF(idsF) = hist3d(binidy+ibin-1, binidx+jbin-1, :);
                end
            end
            F(idF) = sF;
            iblock = iblock + 1;
        end
    end
end

% adjust the negative value caused by accuracy of floating-point
% operations.these value's scale is very small, usually at E-03 magnitude
% while others will be E+02 or E+03 before normalization.
F(F<0) = 0;

% block normalization.
e = 0.001;
l2hysthreshold = 0.2;
fslidestep = nblockw*nblockh*nthet;
switch normmethod
    case 'none'
    case 'l1'
        for fi = 1:fslidestep:size(F,2)
            div = sum(F(fi:(fi+fslidestep-1)));
            F(fi:(fi+fslidestep-1)) = F(fi:(fi+fslidestep-1))/(div+e);
        end
    case 'l1sqrt'
        for fi = 1:fslidestep:size(F,2)
            div = sum(F(fi:(fi+fslidestep-1)));
            F(fi:(fi+fslidestep-1)) = sqrt(F(fi:(fi+fslidestep-1))/(div+e));
        end
    case 'l2'
        for fi = 1:fslidestep:size(F,2)
            sF = F(fi:(fi+fslidestep-1)).*F(fi:(fi+fslidestep-1));
            div = sqrt(sum(sF)+e*e);
            F(fi:(fi+fslidestep-1)) = F(fi:(fi+fslidestep-1))/div;
        end
    case 'l2hys'
        for fi = 1:fslidestep:size(F,2)
            sF = F(fi:(fi+fslidestep-1)).*F(fi:(fi+fslidestep-1));
            div = sqrt(sum(sF)+e*e);
            sF = F(fi:(fi+fslidestep-1))/div;
            sF(sF>l2hysthreshold) = l2hysthreshold;
            div = sqrt(sum(sF.*sF)+e*e);
            F(fi:(fi+fslidestep-1)) = sF/div;
        end
end