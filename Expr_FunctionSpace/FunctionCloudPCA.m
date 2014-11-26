function FunctionCloudPCA(net, layer, stimuli, gridshape, K, varargin)
% function FunctionCloudPCA(net, layer, all_shapes)
%
% for each neuron in layer 'layer' of the NN 'net', this function produces
% a vector which is the representation of that neuron as a function on
% stimulus space, where the matrix stimuli is in the format features x
% num_data_points. It then calculates PCA on all these vectors and
% generates plots of the top K eigenvectors of this function cloud
%
% For each of these K eigenvectors, it plots it as a gridshape(1) x
% gridshape(2) image. Here, it would make most sense to choose translation
% vectors in image space such that you get an equal number of translations
% in x and y, and use a sparse grid so as not to make PCA go crazy
% 
% if layer == 0, this acts on pixels

% Last argument sets the title
if(nargin == 5)
    title_prefix = 'Eigenvector ';
elseif(nargin == 6)
    title_prefix = [varargin{1} ': Eigenvector '];
end

% Feedforward pass of data and PCA on the function cloud representation
data = ExtractDataAtLayer(net, stimuli, layer);
[eigvecs, score, eigvals] = pca(data);

% Generate plots of top K eigenvectors
for i = 1:K
    figure, imagesc(reshape(eigvecs(:,i), gridshape));
    title([title_prefix num2str(i)]);        
end

