function retVal = ApplyDimReduction(X, Y, alg)
% function ApplyDimReduction(X, Y, alg)
% X is M x N in ML format where rows are training examples 
% and columns are features
%
% Y is M x 1 in ML format - a set of labels for each training example
% Y must be whole number valued, or empty (i.e.  [])
% 
% alg is a string to the dimensionality reduction technique to use
%   Options are: 
%   'pca'
%   'tsne'
%   'mds'

if(isempty(Y))
    Y = ones(size(X,1),1);
end

% Apply Dim Reduction Algorithm
switch lower(alg)
    case 'mds'
        % Apply MDS
        D = pdist(X); % Distance vector
        mds = cmdscale(D); % columns are dim reduction vectors
        drData = mds;
        retVal = mds;
    case 'pca'
        [~,score_PCA, latent] = pca(X);
        drData = score_PCA;
%         figure, plot(latent);
%         title('Eigenvalue Spectrum');
        retVal.score_PCA = score_PCA;
        retVal.latent = latent;
    case 'tsne'
        % Want dimensionality reduction to 2
        dim = 3;

        % stopping criteria: number of iterations is no more than 100, runtime is
        % no more than 30 seconds, and the relative tolerance in the embedding is 
        % no less than 1e-3. Taken from Max's tsne example demo_swissroll.m
        opts.maxit = 400; opts.runtime = 900; opts.tol = 1e-3;
        opts.X0 = 1e-5*randn(size(X, 1), dim);

        % Run algorithm
        display('Running t-SNE (Max Code) on Data');
        tic;
        tsne_output = alg_tsne(X, dim, opts);
        drData = tsne_output;
        retVal = tsne_output;
    otherwise 
        display('dim reduction algorithm not supported...')
        return;
end

% Plot 2D
% figure;
% gscatter(drData(:,1), drData(:,2), Y, [], [], 20);
% % legend(Y);
% title([alg ': 2D']);

% Scatter Plot 3D
figure; hold on;
y = unique(Y);
for i = 1:length(y);
    scatter3(drData(Y == y(i),1), drData(Y == y(i),2), drData(Y == y(i),3))
end
% legend(Y)
title([alg ': 3D']);