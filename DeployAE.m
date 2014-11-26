function nn =  DeployAE
%% DeployAE - works with DeepLearnToolbox
% Run the DeepLearnToolbox Autoencoder infrastructure on my datasets

addpath(genpath('./DeepLearnToolbox'))
load ./Data/DigitalManifold-30.mat
X = shapes; Y = targets;

% Choose testing/training data
cutoff = 1100; % size of training set
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

%% (Optional) Increase Data size
addSpeckle = false;
if(addSpeckle)
    noiselevel = 0.01;
    desiredTrainSetSize = 60000;
    desiredTestSetSize = 10000;
    
    % Generate new noisy (and much larger) dataset
    setsize = desiredTrainSetSize + desiredTestSetSize;
    newX = X'; %ML format
    newY = Y';
    ind = randi(size(X,1),1,setsize);
    newX = newX(ind,:) + rand(setsize,size(newX,2)) * noiselevel;
    newY = newY(ind,:);
    
    % Split into training and testing
    train_x = newX(1:desiredTrainSetSize,:);
    train_y = newY(1:desiredTrainSetSize,:);
    test_x = newX(desiredTrainSetSize+1:end,:);
    test_y = newY(desiredTrainSetSize+1:end,:);
end


%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0)
sae = saesetup([900 100]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   100;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);
visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([900 100 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.W{1} = sae.ae{1}.W{1};

% Train the FFNN
opts.numepochs =   1;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.16, 'Too big error');

%% Remove paths
rmpath(genpath('./DeepLearnToolbox'));