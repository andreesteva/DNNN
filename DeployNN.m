%% ADD PATHS
addpath(genpath('ManifoldGenerator'));
addpath(genpath('Data'));

%% Parallelize 
pool = parpool('local', 4);

%% Load Digital Manifold Data

% Load pre-generate digital manifold data
load('./Data/DigitalManifold-30.mat'); % Shapes, targets - 1200 images
X = shapes; % features x examples (N x M)
Y = targets; % classes x examples (C x M)

% Choose testing/training data
cutoff = 1100;
ind = randperm(size(X,2));
train_ind = ind(1:cutoff);
test_ind = ind(cutoff+1:end);
train_x = X(:,train_ind); train_y = Y(:, train_ind);
test_x = X(:, test_ind); test_y = Y(:, test_ind);

 %% Load MNIST
load('./DeepLearnToolbox/data/mnist_uint8.mat'); 
% train_x, train_y, test_x, test_y
%  ML format for all of them

% Format into inverse-ML for NN Toolbox
train_x = double(train_x');
train_y = double(train_y');
test_x = double(test_x');
test_y = double(test_y');

%% Train NN

% Architecture & Training Function
layers = [];
net = feedforwardnet(layers, 'trainscg'); % Change to cross entropy objective with newer matlab version

% Output Transfer Function
net.layers{end}.transferFcn = 'softmax'; % Also, 'purelin' is the identify function and can work well

% Performance Function
net.performFcn = 'crossentropy'; % use 'mse' or 'crossentropy'
net.performParam.regularization = 0.0; % msereg = (1-r)*mse + r*msw,
net.trainParam.goal = 0.001; % Mean squared error of 0.001 is performance metric

% Pre & Post Processing Functions
net.inputs{1}.processFcns = {'mapminmax'}; % Remove 'mapminmax' pre-process function
net.outputs{end}.processFcns = {'mapminmax'}; % Remove 'mapminmax' post-process function

% Training Parameters
net.trainParam.epochs = 50000;
net.divideFcn = '';
net.divideFcn = 'dividerand'; %randomly divide data up into train/test/validation sets
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Train
display(['Training NN with layers = ' num2str(layers)]);
if(exist('pool', 'var'))
    display('Parallel Training');
    [netT,tr] = train(net,train_x, train_y, 'useParallel', 'yes', 'showResources','yes');
else
    display('Serial Training');
    [netT,tr] = train(net,train_x, train_y);
end
netT.userdata = tr;

% Check training error

trainingError = TestNN(netT, train_x, train_y)

% Check testing error
testingError = TestNN(netT, test_x, test_y)

    %% Save NN
save(['TNet-' num2str(length(layers)) 'HL-' num2str(layers), ...
    '-TrSetSize=' num2str(size(train_x,2)) '-Epoch=' num2str(net.trainParam.epochs)], ...
    'netT', 'tr', 'train_x', 'train_y', 'test_x', 'test_y');

































   

