%%  Run a CNN using the DeepLearnToolbox

%% Create Data
addpath(genpath('DeepLearnToolbox'));
addpath(genpath('Data'));

% Load pre-generate digital manifold data
load('./Data/DigitalManifold-28.mat'); % Shapes, targets - 1200 images

% Choose testing/training data
ind = randperm(size(shapes,2));
cutoff = 1100;
train_ind = ind(1:cutoff);
test_ind = ind(cutoff+1:end);
train_x = shapes(:, train_ind); train_y = targets(:, train_ind);
test_x = shapes(:, test_ind); test_y = targets(:, test_ind);

% Reshape it
train_x = reshape(train_x, 28,28,[]);
test_x = reshape(test_x, 28,28,[]);

%% Train a CNN

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};


opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 2000;

cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);

% Training Error
[trainingError, trainingMistakes] = cnntest(cnn, train_x, train_y);
trainingError

% Testing Error
[testingError, testingMistakes] = cnntest(cnn, test_x, test_y);
testingError

%plot mean squared error
figure; plot(cnn.rL); 
title(['MSE vs Batch #']);






