%% Deblurring via Cauchy proximal splitting algorithm
% Y = HX + N
% Y is the blurred and noisy image
% X is the clear image (object of interest)
% H is the point spread function (blurring operation)
% N is the additive zero-meam Gaussian noise with variance \sigma^2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Some Important Variables
%       ** X: Noise-free Image.
%
%       ** filterSize: The size of the blurring filter.
%
%       ** BSNRdb: Blurred-SNR value in decibels.
%
%       ** maxIter: Maximum number of FB iterations.
%
%       ** Y: Noisy and blurred image.
%
%       ** mu: CPS step size
%
%       ** gamma: Cauchy scale parameter
%
%       ** errorCriterion: The stopping criterion for the FB algorithm.
%
%       ** x_hat: The reconstructed image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LICENSE
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
% 
% Copyright (C) Oktay Karakus,PhD 
% University of Bristol, UK
% o.karakus@bristol.ac.uk
% April 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% REFERENCE
%
% [1] O Karakus, P Mayo, and A Achim. "Convergence Guarantees for 
%     Non-Convex Optimisation with Cauchy-Based Penalties"
%       arXiv preprint.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars
close all
clc
imName = 'cameraman';
X = double(imread([imName '.tif']));
filterSize = 5;
h = fspecial('gaussian',[filterSize filterSize], 1);
blurred = imfilter(X, h, 'circular');
BSNRdb = 40; % Blurred-signal to noise ratio in decibels
sigma = norm(blurred-mean(mean(blurred)),'fro')/sqrt(size(blurred, 1)*size(blurred, 2)*10^(BSNRdb/10)); % noise std calculation
Y = blurred + sigma*randn(size(blurred)); % The noisy and blurred image Y.
K  = psf2otf(h, size(X));
KC = conj(K);
A  = @(x) real(ifft2(K.*fft2(x)));      % Forward operator
AH = @(x) real(ifft2(KC.*fft2(x)));     % Inverse operator
grad_f_x = @(x) AH(A(x) - Y);           % Gradient operator
xx = ones(size(Y));
yy = 0*ones(size(Y));
Lip = norm(grad_f_x(xx) - grad_f_x(yy), 2)/norm(xx - yy, 2); % A general 
%           calculation for Lipschitz constant via gradiend Lipschitz 
%           definition of data fidelity term.
mu = 1.5/Lip; 
% error parameter
PSNR_noisy = psnr(uint8(Y), uint8(X));
RMSE_noisy = sqrt(mean((Y - X).^2));
delta_x = inf;
x_hat = zeros(size(Y));
iter = 1;
maxIter = 250;
errorCriterion = 1e-3;
old_X = x_hat;
gamma = 17*sqrt(mu); % The lower bound for guaranteeing convergence sqrt(mu)/2
tic;
while (delta_x(iter) > errorCriterion) && (iter < maxIter)
    iter = iter + 1;
    Z = x_hat - (mu)*(grad_f_x(x_hat));
    x_hat = CauchyProx(Z, gamma, mu);
    delta_x(iter) = max(abs( x_hat(:) - old_X(:) )) / max(abs(old_X(:))); % Error calculation
    old_X = x_hat;
end
timeSim = toc;
PSNR_regularized = psnr(uint8(x_hat), uint8(X));
RMSE_regularized = sqrt(mean((x_hat(:) - X(:)).^2));
figure;
set(gcf, 'Position', [100 100 900 400])
subplot('Position', [0.0101, 0.05001, 0.3, 0.95])
imshow(uint8(X));       % Original
title('Original Image')
subplot('Position', [0.3401, 0.05001, 0.3, 0.95])
imshow(uint8(Y));       % Blurred
title(['Blurred Image (PSNR = ' num2str(PSNR_noisy) ' dB)'])
subplot('Position', [0.6701, 0.05001, 0.3, 0.95])
imshow(uint8(x_hat));   % Cauchy Reconstructed
title(['CPS Reconstructed (PSNR = ' num2str(PSNR_regularized) ' dB)'])
fprintf('Cauchy proximal splitting (CPS) for image deblurring\nSolved after %d iterations in %.3f seconds\nNoisy PSNR = %.3f dB\nReconstructed PSNR = %.3f dB\n', iter, timeSim, PSNR_noisy, PSNR_regularized)