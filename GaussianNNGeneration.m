function netsOut = GaussianNNGeneration(netsIn, amount)
% 
% function nets = GaussianNNGeneration(nets0)
% 
% input a set of identical neural nets in the cell nets0, and
% enter a postiive integer amount
% 
% this function will generate that many neural nets with the same
% architecture but whose weights/biases have been drawn from a gaussian
% distribution whose mean and std are the mean and std of the weights of
% the neural nets in nets0. The means/stds of each layer are calculated
% from that layer only.
% 

WHENCE - THIS IS ALL UNFINISHIED

display('Gathering mean and std of input weights');
netsOut = {};

exnet = netsIn{1};
opts.numNN = amount;
netsOut = CreateNN(opts); % creates 'amount' neural nets

weights1 = {};
weights2 = {};
for i = 1:length(netsIn)
    if(nets{i}.userdata.genError < err)
        nets0{end+1} = nets{i};        
        weights1{end+1} = nets{i}.IW{1,1};
        weights2{end+1} = nets{i}.LW{2,1};        
    end
end
m1 = mean(mean(cell2mat(weights1)));
s1 = mean(std(cell2mat(weights1)));
m2 = mean(mean(cell2mat(weights2)));
s2 = mean(std(cell2mat(weights2)));

% Generate neural nets with weights taken from gaussian distribution 
display('generate new neural nets - normally distributed');
netsG = {};
opts.layers = nets0{1}.layers{1}.size; % 100, in this case
for i = 1:length(nets0)    
    n = CreateNN(opts);
    n.input.exampleInput = shapes(:,1);
    n.IW{1,1} = m1 + s1*randn(size(nets0{i}.IW{1,1}));% Gaussian with mean m and std s
    n.LW{2,1} = m2 + s2*randn(size(nets0{i}.LW{2,1}));% Gaussian with mean m and std s
    netsG{i} = n;
    if(mod(i,25) == 0), display(num2str(i)); end;
end