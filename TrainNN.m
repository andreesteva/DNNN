function [netT] = TrainNN(X,Y,opts)
% function [netT] = TrainNN(X,Y,opts)
% couples with CreateNN to create and train a neural network from the
% MATLAB toolbox
%
% X and Y are in inverse ML format
% opts can have the following fields
% if(isfield(opts, 'testRatio')), testRatio = opts.testRatio; else testRatio = 0.10; end
%
% From CreateNN:
% if(isfield(opts, 'layers')), layers = opts.layers; else layers = 100; end
% if(isfield(opts, 'epochs')), epochs = opts.epochs; else epochs = 2500; end
% if(isfield(opts, 'max_fail')), max_fail = opts.max_fail; else max_fail = 6; end
% if(isfield(opts, 'numNN')), numNN = opts.numNN; else numNN = 1; end
% if(isfield(opts, 'postprocessFcns')), postprocessFcns = opts.postprocessFcns; else postprocessFcns = {}; end
% if(isfield(opts, 'trainRatio')), trainRatio = opts.trainRatio; else trainRatio = 0.85; end
% if(isfield(opts, 'valRatio')), valRatio = opts.valRatio; else valRatio = 0.15; end
% if(isfield(opts, 'performanceGoal')), performanceGoal = opts.performanceGoal; else performanceGoal = 10^-6; end

    
% Options
if(isfield(opts, 'testRatio')), testRatio = opts.testRatio; else testRatio = 0.10; end

% Choose testing/training data
ind = randperm(size(X,2));
cutoff = round(((1-testRatio) * size(X,2)));
train_ind = ind(1:cutoff);
test_ind = ind(cutoff+1:end);
train_x = X(:,train_ind); train_y = Y(:, train_ind);
test_x = X(:, test_ind); test_y = Y(:, test_ind);

% Create Neural Network
net = CreateNN(opts);

% Train
if(~isempty(gcp('nocreate')))
    display('Parallel Training');
    [netT,tr] = train(net,train_x, train_y, 'useParallel', 'yes');
else
    [netT,tr] = train(net,train_x, train_y);
end    

netT.userdata.tr = tr;
netT.userdata.genError = TestNN(netT, test_x, test_y);
netT.userdata.trainError = TestNN(netT, train_x, train_y);   
