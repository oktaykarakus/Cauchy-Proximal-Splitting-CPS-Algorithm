%% Deblurring via Cauchy proximal splitting algorithm
% y = x + n
% y is the 1D noisy signal
% x is the clear (noise-free) signal (object of interest)
% n is the additive zero-meam Gaussian noise with SNR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Some Important Variables
%       ** sizeSignal: Exponent of 2 for the size of the signal in time domain.
%
%       ** M: The size of the signal in time domain.
%
%       ** N: The size of the signal in frequency domain.
%
%       ** SNRdB: Noise SNR value in decibels.
%
%       ** Niter: Maximum number of FB iterations.
%
%       ** x: Noise-free signal
%
%       ** y: Noisy signal
%
%       ** mu: FB step size
%
%       ** gamma: Cauchy scale parameter
%
%       ** x_hat: The reconstructed signal in frequency domain.
%
%       ** x_Cauchy: The reconstructed signal in frequency domain.
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
%% Parameter Initialization
sizeSignal = 7;
M = 2^sizeSignal;
N = 2^(sizeSignal + 2);
SNRdB = 3;
rmse = @(err) sqrt(mean(abs(err(:)).^2));
truncate = @(x, M) x(1:M);
AH = @(x) fft(x, N)/sqrt(N);
A = @(X) truncate(ifft(X), M) * sqrt(N);
Niter = 500;
[x,y] = wnoise(3, sizeSignal, SNRdB);
x = x';
y = y';
%% Cauchy
x_hat = AH(zeros(size(y))); % regularized result
iter = 1;
old_X = x_hat;
grad_f_x = @(x) AH(A(x) - y); % gradient operator
xx = ones(size(y));
yy = 0*ones(size(y));
Lip = norm(grad_f_x(xx) - grad_f_x(yy), 2)/norm(xx - yy, 2); % A general calculation for Lipschitz constant.
mu = 1.5/Lip;
gamma = 2*sqrt(mu)/2;
delta_x = inf;
tic;
while (delta_x(iter) > 1e-3) && (iter < Niter)
    iter = iter + 1;
    Z = x_hat - mu*(AH(A(x_hat) - y));
    x_hat = CauchyProx(real(Z), gamma, mu);
    delta_x(iter) = max(abs( x_hat(:) - old_X(:) )) / max(abs(old_X(:))); % Error calculation
    old_X = x_hat;
end 
x_Cauchy = A(x_hat);
timeSim = toc;
RMSE_noisy = rmse(x - y);
RMSE_regularized = rmse(x - x_Cauchy);
fprintf('Cauchy proximal splitting (CPS) for 1D denoising\nSolved after %d iterations in %.3f seconds\nNoisy RMSE = %.3f\nReconstructed RMSE = %.3f\n', iter, timeSim, RMSE_noisy, RMSE_regularized)

figure;
set(gcf, 'Position', [100 100 800 300])
subplot('Position', [0.0501, 0.1001, 0.9, 0.85])
plot(x, 'b', 'Linewidth', 1.5)
hold on
plot(y, 'k-.', 'Linewidth', 1)
plot(x_Cauchy, 'r--', 'Linewidth', 2)
grid on
legend('Noise-free', ['Noisy (SNR = ' num2str(SNRdB) ' dB)'], 'CPS')
text(40, 0.9*max(y), ['RMSE_{Noisy} = ' num2str(RMSE_noisy)], 'Color', 'Black')
text(40, 0.6*max(y), ['RMSE_{CPS} = ' num2str(RMSE_regularized)], 'Color', 'Red')