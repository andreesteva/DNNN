function [net, tr] = Expr_LineSegments_TrainNN(shapes, targets, architecture)
% Solve a Pattern Recognition Problem with a Neural Network
% Script generated by Neural Pattern Recognition app
% Created Wed Oct 15 16:36:21 PDT 2014
%
% This script assumes these variables are defined:
%
%   shapes - input data.
%   targets - target data.
%
% Calling Options:
% function [net, tr] = Expr_LineSegments_TrainNN(shapes, targets)
% function [net, tr] = Expr_LineSegments_TrainNN(shapes, targets, filename)
%
% if filename is defined, the function create a new function file which
% simulates the functionality of the net. It will name is 'filename'


x = shapes;
t = targets;

% Create a Pattern Recognition Network
hiddenLayerSize = architecture;
net = patternnet(hiddenLayerSize);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'mapminmax'};
net.output.processFcns = {'mapminmax'};
% net.input.processFcns = {};
% net.output.processFcns = {};


% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% For help on training function 'trainscg' type: help trainscg
% For a list of all training functions type: help nntrain
net.trainFcn = 'trainscg';  % Scaled conjugate gradient

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
% net.performFcn = 'crossentropy';  % Cross-entropy
net.performFcn = 'mse';  % Cross-entropy

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit'};


% Train the Network
if(~isempty(gcp('nocreate')))
    display('Parallel Training');
    [net,tr] = train(net,x,t, 'useParallel', 'yes');
else
    [net,tr] = train(net,x,t);
end    

tr.genError = TestNN(net, x(:, tr.testInd), t(:, tr.testInd));
tr.trainError = TestNN(net, x(:, tr.trainInd), t(:, tr.trainInd));  
tr.valError = TestNN(net, x(:, tr.valInd), t(:, tr.valInd));  
tr.totalError = TestNN(net, x, t);


% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)
%figure, ploterrhist(e)


end

function err = TestNN(net, X, Y)
    err = sum(sum(abs(compet(net(X)) - Y)))/2 /size(X,2);
end
