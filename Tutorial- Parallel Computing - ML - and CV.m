%% Parallel Computing

% Create a pool of 2 workers (one on each core)
parpool('local', 2);

% Delete current pool
delete(gcp);

% Other functions
%parfor
 %   batch
  %  distributed

% mx'' + bx' + kx = 0
% convert for into parfor
% use pool of MATLAB workers
