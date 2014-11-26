%% Task Complexity Experiment
% Corresponding evernote file - https://www.evernote.com/shard/s199/sh/c2120cc1-5e60-4c4f-b107-dbafe1afa05d/fe18f92aeb9b8408a0379a77a5c89a80
% It has the instructions and thought process employed here
%

%% Blank Slate
clc
clear all

%% Use a 4 shape manifold
load('./Data/FiveShapeManifold-30.mat');

%% Train a few hundred neural nets 

numNN = 200;
opts.layers = 20;

% Train the NNs
nets = {};
f = figure; 
title('EigenAnalysis Update')
for i = 1:numNN
    display(['Training Net ' num2str(i)]);
    [netT] = TrainNN(shapes, targets, opts);
    nets{end+1} = netT;
    
    % Progressive Save and Update of Eigenanalysis plot
    if(mod(i,20) == 0)
        save([num2str(i) ' Trained 1HL NN'], 'nets')
        ind = randperm(length(shapes));
        m = EigenAnalysis(nets, shapes(:,ind(1:1000)), targets(:,ind(1:1000)), 1);
        figure(f); plot(m(1:50))
        title(['EigenAnalysis Update - ' num2str(i) '/' num2str(numNN) ' neural nets']);
        ylabel('Mean Correlation');
        xlabel('Eigenvector Index');
    end
    
end

%% Save Nets
save([num2str(length(nets)) ' Trained 1HL NN - FiveShapeManifold'], 'nets', '-v7.3');

%% Test nets
for i =1:length(nets)   
    TestNN(nets{i}, shapes, targets)
end

%% Compare 3 shapes data to 4 shapes data

% Calculate eigenanalysis of 3 shapes:
load('./Data/Trained on 3 shapes/200 Trained 1HL NN - AnalogManifoldnets.mat', 'nets');
load('./Data/AnalogManifold-30.mat', 'shapes', 'targets');
ind = randperm(length(shapes));
evd3 = EigenAnalysis(nets, shapes(:,ind(1:1000)), targets(:,ind(1:1000)), 1);

% Calculate eigenanalysis of 4 shapes
load('./Data/Trained on 4 shapes/200 Trained 1HL NN - FourShapeManifold.mat', 'nets');
load('./Data/FourShapeManifold-30.mat', 'shapes', 'targets');
ind = randperm(length(shapes));
evd4 = EigenAnalysis(nets, shapes(:,ind(1:1000)), targets(:,ind(1:1000)), 1);

% Calculate eigenanalysis of 5 shapes
load('./Data/Trained on 5 shapes/200 Trained 1HL NN - FiveShapeManifold.mat', 'nets');
load('./Data/FiveShapeManifold-30.mat', 'shapes', 'targets');
ind = randperm(length(shapes));
evd5 = EigenAnalysis(nets, shapes(:,ind(1:1000)), targets(:,ind(1:1000)), 1);

%%
figure; hold on;
plot(evd3(1:50), 'b');
plot(evd4(1:50), 'r');
plot(evd5(1:50), 'g');
legend('3 shapes', '4 shapes', '5 shapes');
title('EigenAnalysis');
ylabel('Mean Correlation');
xlabel('Eigenvector Index');
makeFiguresPretty