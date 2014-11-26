% DeployDNN - Using DeepLearnToolbox
addpath(genpath('../DeepLearnToolbox'))

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

% Convert to ML format
train_x = train_x';
train_y = train_y';
test_x = test_x';
test_y = test_y';

%% Train a DNN

rand('state',0)
nn                      = nnsetup([900 100 3]);  % no HL   
nn.output               = 'softmax';                   %  use softmax output
opts.numepochs          = 100;                           %  Number of full sweeps through data
opts.batchsize          = 100;                        %  Take a mean gradient step over this many samples
opts.plot               = 1;                           %  enable plotting
nn = nntrain(nn, train_x, train_y, opts);                %  nntrain takes validation set as last two arguments (optionally)

[er, bad] = nntest(nn, test_x, test_y);
er
assert(er < 0.1, 'Too big error'); 