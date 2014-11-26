function netsOut = CreateNN(opts)

% Options
if(isfield(opts, 'layers')), layers = opts.layers; else layers = 100; end
if(isfield(opts, 'epochs')), epochs = opts.epochs; else epochs = 2500; end
if(isfield(opts, 'max_fail')), max_fail = opts.max_fail; else max_fail = 6; end
if(isfield(opts, 'numNN')), numNN = opts.numNN; else numNN = 1; end
if(isfield(opts, 'postprocessFcns')), postprocessFcns = opts.postprocessFcns; else postprocessFcns = {}; end
if(isfield(opts, 'trainRatio')), trainRatio = opts.trainRatio; else trainRatio = 0.85; end
if(isfield(opts, 'valRatio')), valRatio = opts.valRatio; else valRatio = 0.15; end
if(isfield(opts, 'performanceGoal')), performanceGoal = opts.performanceGoal; else performanceGoal = 10^-6; end

netsOut = {};
for i = 1:numNN
        % Architecture & Training Function
    %     layers = 100;
    net = feedforwardnet(layers, 'trainscg'); % Change to cross entropy objective with newer matlab version

    % Output Transfer Function
    net.layers{end}.transferFcn = 'softmax'; % Also, 'purelin' is the identify function and can work well

    % Performance Function
    net.performFcn = 'crossentropy'; % use 'mse' or 'crossentropy'
    net.performParam.regularization = 0.0; % msereg = (1-r)*mse + r*msw,
    net.trainParam.goal = performanceGoal; % Mean squared error of 0.000 is performance metric which will stop the sim

    % Pre & Post Processing Functions
    net.inputs{1}.processFcns = {}; %  {'mapminmax'} Remove 'remove constant rows' pre-process function & mapminmax
    net.outputs{end}.processFcns = postprocessFcns; % i.e. 'mapminmax'

    % Training Parameters
    net.trainParam.epochs = epochs;

    %     net.divideFcn = '';
    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'sample';  % Divide up every sample
    net.trainParam.max_fail = max_fail; % default is 6 validation fails before exiting - a validation fail means that the validation error did not decrease that epoch
    net.divideParam.trainRatio = trainRatio;
    net.divideParam.valRatio = valRatio;
    net.divideParam.testRatio = 0;
    
    netsOut{end+1} = net;
end

% Return a NN structure if we only wanted one nn, otherwise return a cell
% array
if(length(netsOut) == 1)
    netsOut = netsOut{1};
end