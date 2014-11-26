%% Custom Netowrk via http://www.mathworks.com/help/nnet/ug/create-and-train-custom-neural-network-architectures.html#bss4gz0-18

% Network definition and architecture
net = network;
net.numInputs = 2;
net.numLayers = 3;

% Bias Connections
net.biasConnect = [1; 0; 1];

% Input and Layer Connections
net.inputConnect = [1 0; 1 1; 0 0];
net.layerConnect = [0 0 0; 0 0 0; 1 1 1];

% Output Connections
net.outputConnect = [0 1 1];

% Inputs
net.inputs{1}.exampleInput = [0 10 5; 0 3 10];
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.inputs{2}.size = 5;

% Layers
net.layers{1}.size = 4;
net.layers{1}.transferFcn = 'tansig';
net.layers{1}.initFcn = 'initnw';
net.layers{2}.size = 3;
net.layers{2}.transferFcn = 'logsig';
net.layers{2}.initFcn = 'initnw';
net.layers{3}.initFcn = 'initnw';
net.layers{3}.size = 1; % not in documentation

% Outputs

% Biases, Input Weights, and Layer Weights
net.inputWeights{2,1}.delays = [0 1]; % the input weights from input 1 to layer 2 are set to [0 1]. Its a Tapped Delay Line
net.inputWeights{2,2}.delays = 1;
net.layerWeights{3,3}.delays = 1; % layerWeights{3,3} reads "the weights from layer 3 to layer 3

% Network Functions
net.initFcn = 'initlay';
net.performFcn = 'mse';
net.trainFcn = 'trainlm';
net.divideFcn = 'dividerand'; % divide training data randomly into training, test, and validation. 
                                %Network is trained on training data until its performance begins to decrease on the validation data, 
                                % which signals that generalization has peaked
net.plotFcns = {'plotperform','plottrainstate'};

% Initialization
net = init(net);

% Training
X = {[0; 0] [2; 0.5]; [2; -2; 1; 0; 1] [-1; -1; 1; 0; 1]};
T = {[1; 1; 1] [0; 0; 0]; 1 -1};

% Output before Training
Y = sim(net,X)

% Training
net = train(net,X,T);

% Output after training
Y = sim(net,X)

