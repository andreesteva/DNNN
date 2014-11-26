function m = EigenAnalysis(nets, X,Y, L)
% X and Y are the data, in inverse ML format
% L is the layer from which to extract data from each network in the cell
% variable 'nets'. nets{i} must contain a neural network trained by the
% MATLAB NN toolbox
%
%   1. Take the data and...
%   2. Pass it through each network nets{i} and extract it at layer L (Activation i)
%   3. Take all activations of network nets{i} & correlate them into a matrix
%   4. take the EVD of that matrix
%   5. Pick the jth eigenvector, and pairwise correlate it across network matrices i to get a correlation vector C
%   6. Take the mean of C, to get a point on the plot above

numdatapts = size(X,2);
display('Running EigenAnalysis');
tic

% Gather Eigenvectors of the correlation matrix across neural nets

Vn = zeros(numdatapts, numdatapts, length(nets));
for n = 1:length(nets)
    a = ExtractDataAtLayer(nets{n}, X,Y,L);
    if(L==2), a = compet(a); end
    C = corr(a); C = bsxfun(@max,C,C');

    % Keep eigenvectors of correlation matrix
    [V,D] = eig(C);
    d = diag(D);
    [d,ind] = sort(d, 'descend'); %sort in descending order
    V = V(:,ind);
    Vn(:,:,n) = V;
end

toc

% Compare fixed eigenvectors of the activation-correlation matrix across neural nets
for e = 1:numdatapts; 
    Ve = squeeze(Vn(:,e,:));
    Cv = max( corr(Ve,Ve), corr(Ve,-Ve)); 
    Cv = max(Cv,Cv');
    acpwevcNN{e} = Cv;
end

toc

% Average acpwevcnNN vs eigenvector index
m = [];
v = [];
for i = 1:length(acpwevcNN)
    m = [m mean(mean(acpwevcNN{i}))];
    v = [v var(acpwevcNN{i}(:))];
end
% figure;
% plot(m)
% % figure, errorbar(1:length(acpwevcNN), m, v); 
% title('Average Activation-correlation pairwise eigenvector max-correlation across NNs')
% xlabel('Eigenvector index');
% ylabel('Mean Correlation');

toc