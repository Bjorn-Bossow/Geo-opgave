function [g_low,g_high] = Gauss2Dconv_v1(lon,lat,val,sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function that applies a 2D Gaussian filter to a grid by a convolution in
% the Fourier domain. Output is both the low- and high-pass filtered grid.
%
% -------------------------------------------------------------------------
% 
% Input:
%   lon     n x m array of longitude [deg]
%   lat     n x m array of latitude [deg]
%   val     n x m array of signal values to be filtered
%   sigma   scalar that determines the filter width [m]
%
% Output:
%   g_low   n x m array of low-pass filtered values
%   g_high  n x m array of high-pass filtered values, i.e. val-g_low
%
% -------------------------------------------------------------------------
% Author: Tim Jensen (DTU) 10/10/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Conversion factor
deg2rad = pi/180;
R       = 6371000; % [m]

% Get dimensions
[n,m] = size(val);

% Check that grid is consistent
if size(lon,1) ~= n
    error('Grid dimensions not consistent');
elseif size(lon,2) ~= m
    error('Grid dimensions not consistent');
elseif size(lat,1) ~= n
    error('Grid dimensions not consistent');
elseif size(lat,2) ~= m
    error('Grid dimensions not consistent');
end

% Derive grid increments
dinc = lon(1,2) - lon(1,1);

% Convert from degrees to meters
sample_interval = dinc*deg2rad*R;

% Transform sigma from meters to degrees
sigma = sigma / sample_interval;

% Account for NaN values
val( isnan(val) ) = 0;


%% Form 2D Fourier Transform

% Form 2D Fourier transform
F = fft2(val);

% Get dimensions
[N,M] = size(F);

% Derive frequency
k = [0:ceil(N/2)-1,-floor(N/2):-1]/N;
l = [0:ceil(M/2)-1,-floor(M/2):-1]/M;
[l,k] = meshgrid(l,k);

% Form variables with zero frequency at center
F_center = fftshift(F);
k_center = fftshift(k);
l_center = fftshift(l);

% Visualize
% plot_Fkl(F,k,l);
% plot_Fkl(F_center,k_center,l_center);


%% Form Filter Kernel

% Form kernel
H_center = exp( -sigma^2 * ( k_center.^2 + l_center.^2 ) / 2 );

% Visualize kernel
% plot_Fkl(H_center,k_center,l_center);


%% Apply Filter and Reconstruct Signal

% Apply filter
G_center = H_center .* F_center;
G = ifftshift(G_center);

% Form inverse FFT
g_low  = ifft2(G);

% Derive high-pass product
g_high = val - g_low;