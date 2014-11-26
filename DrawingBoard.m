[w1,w2] = meshgrid(-200:0.2:200, -20:0.2:20);
W = [w1(:), w2(:)];
% x1 = 2*rand(2,1)-1; x1 = x1/norm(x1); 
% x2 = 1./x1; x2(1) = -x2(1); x2 = x2;
x1 = [1;0];
x2 = [0; 10];
t1 = 0;
t2 = 1;
E1 = 1/2*(t1 - reshape(W*x1, size(w1))).^2;
E2 = 1/2*(t2 - reshape(W*x2, size(w1))).^2;
surf(w1,w2,E1 + E2)
%% Pick a shape, and observe how that shape moves around in input space and activation space
circles = shapes(:, 1:length(shapes)/3);
t = targets(:, 1:length(shapes)/3);
[~,Y] = max(t); 
pcaOut_input = ApplyDimReduction(circles', [], 'pca');

activation = ExtractDataAtLayer(nets{1}, circles, t, 1);
pcaOut_activation = ApplyDimReduction(activation', [], 'pca');

output = nets{1}(circles);
pcaOut_output = ApplyDimReduction(output', [], 'pca');

    %% plot individually
    range = [1:153, 20:153:153*153];
    range = 1:length(circles);
    doit = @(s) scatter3(s.score_PCA(range, 1), s.score_PCA(range, 2), s.score_PCA(range, 3));
    figure, doit(pcaOut_input);
    figure, doit(pcaOut_activation);

%% Correlation of the weights of analog-trained NNs
vectorizeLW = @(c) c.IW{1}(:);
weights = cell2mat(cellfun(vectorizeLW, nets, 'UniformOutput', false));
c = corr(weights);
figure, imagesc(c); colorbar;
title('Correlation of Input weights of 100 NNs trained to 0 error on AnalogManifold-30');
makeFiguresPretty;

%% Gather Eigenvectors of the correlation matrix across neural nets, activating in a bootstrapped way
L = 1; % layer    
sampleSize = 2000;
Vn = zeros(sampleSize,sampleSize, length(nets0));
r = randperm(length(shapes));
x = shapes(:,r(1:sampleSize));
t = targets(:,r(1:sampleSize));
for n = 1:length(nets0)
    n
    a = ExtractDataAtLayer(nets0{n}, x,t,L);
    if(L==2), a = compet(a); end
    C = corr(a); C = bsxfun(@max,C,C');

    % Keep eigenvector e of correlation matrix
    [V,D] = eig(C);
    d = diag(D);
    [d,ind] = sort(d, 'descend'); %sort in descending order
    V = V(:,ind);
    Vn(:,:,n) = V;
end
    %% Compare fixed eigenvectors of the activation-correlation matrix across neural nets
    tic;
    for e = 1:sampleSize; % eigenvector to compare
        display(['Calculating Pair-wise eigenvector correlations across NNs for eigenvector ' num2str(e)]);
        Ve = squeeze(Vn(:,e,:));
        Cv = max( corr(Ve,Ve), corr(Ve,-Ve)); 
        Cv = max(Cv,Cv');
        acpwevcNN{e} = Cv;
    end
    %% Average acpwevcnNN vs eigenvector index
    m = [];
    v = [];
    for i = 1:length(acpwevcNN)
        m = [m mean(acpwevcNN{i}(:))];
        v = [v var(acpwevcNN{i}(:))];
    end
    figure, errorbar(1:length(acpwevcNN), m, v); 
    title('Average Activation-correlation pairwise eigenvector max-correlation across NNs')
    xlabel('Eigenvector index');
    ylabel('Mean Correlation');

%% Train Many identical Autoencoders (with MNN)
% load('Data/DigitalManifold-30.mat'); %shapes, targets
load('Data/AnalogManifold-30.mat');
numNN = 1;
opts.layers = 100;
opts.epochs = 10000;
AEs = {};

% Train the NNs
nets = {};
for i = 1:numNN
    display(['Training AE ' num2str(i)]);
    [netT] = TrainAE(shapes, opts);
    AEs{end+1} = netT;
    if(mod(i,25) == 0)
        save([num2str(i) ' Trained AEs'], 'AEs')
    end
end
save([num2str(length(AEs)) ' Trained AEs', 'AEs']);

%% Null Hypothesis
% works on set of 1HL neural nets in variable 'nets'

% Calculate mean and std of the weights in the first layer
weightSize = size(nets{1}.IW{1});
stack = zeros([weightSize length(nets)]);
for i = 1:length(nets)
    stack(:,:,i) = nets{i}.IW{1};
end
m = mean(stack(:));
s = std(stack(:));

% Replace weights with random gaussian values around this mean and std
nets_random = nets;
for i = 1:length(nets_random)    
    nets_random{i}.IW{1} = m + s*randn(weightSize);
end

    %% Take 1HL NNs, make IWs normally random, freeze them, and train the layer weights
    netR = nets_random{1};
    shapesAct = ExtractDataAtLayer(netR, shapes, targets, 1);
    opts.layers = [];
    genErrs = [];
    trainErrs = [];
    for i = 1:100
        netTop = TrainNN(shapesAct,targets,opts);
        genErrs = [genErrs, netTop.userdata.genError];
        trainErrs = [trainErrs, netTop.userdata.trainError];
    end
    mean(genErrs)
    mean(trainErrs)
    
%% Generate random NNs from the mean and std of the weights of trained ones
% display('loading data');
load('./Data/Trained on digital/Net1/500 Trained 1HL NNnets.mat'); % load nets
% load('./Data/DigitalManifold-30.mat');

% Gather mean and std of input weights across low-error neural nets
display('Gathering mean and std of input weights');
err = 0.01;
nets0 = {};
weights1 = {};
weights2 = {};
bias1 = {};
bias2 = {};
for i = 1:length(nets)
    if(nets{i}.userdata.genError < err)
        nets0{end+1} = nets{i};        
        weights1{end+1} = nets{i}.IW{1,1};
        weights2{end+1} = nets{i}.LW{2,1};
        bias1{end+1} = nets{i}.b{1};
        bias2{end+1} = nets{i}.b{2};
    end
end

weights1 = cell2mat(weights1);
weights2 = cell2mat(weights2);
bias1 = cell2mat(bias1);
bias2 = cell2mat(bias2);

m1 = mean(weights1(:)); s1 = std(weights1(:)); % note this bug - it could be the reason my results weren't significant: s1 = mean(std(cell2mat(weights1)));
m2 = mean(weights2(:)); s2 = std(weights2(:)); % s2 = mean(std(cell2mat(weights2)));
m1b = mean(bias1(:));   s1b = std(bias1(:));
m2b = mean(bias2(:));   s2b = std(bias2(:));

% Generate neural nets with weights taken from gaussian distribution 
display('generate new neural nets - normally distributed');
netsG = {};
opts.layers = nets0{1}.layers{1}.size; % 100, in this case
for i = 1:length(nets0)        
    n = CreateNN(opts);
    n.input.exampleInput = shapes(:,1);
    n.output.exampleOutput = targets(:,1);
    n.IW{1,1} = m1 + s1*randn(size(nets0{i}.IW{1,1}));% Gaussian with mean m and std s
    n.LW{2,1} = m2 + s2*randn(size(nets0{i}.LW{2,1}));% Gaussian with mean m and std s
    n.b{1} = m1b + s1b*randn(size(nets0{i}.b{1}));
    n.b{2} = m2b + s2b*randn(size(nets0{i}.b{2}));
    netsG{i} = n;
    if(mod(i,25) == 0), display(num2str(i)); end;
end
save('Null Hypothesis untrained NNs - to within 1 percent error', 'netsG');

%% Calculate the maxcorr of eigenvectors of activation correlation matrices across neural nets
% load('./Data/DigitalManifold-30.mat');
% load('./Data/Trained on digital/Net1/Null Hypothesis,... untrained NNs - to within 1 percent error.mat'); %load netsG

nets_trial = netsG; 
% nets_trial = nets0;

% Convert each neural net into an activation-correlation-matrix & 
% Do eigenvector decomposition on each matrix
display('Embed NN into activation-correlation-matrix space & do eigenvector decomposition');
L = 1; % layer    
Vn = zeros(size(shapes,2) , size(shapes,2), length(nets_trial));
for n = 1:length(nets_trial)
    display([' Embedding NN ' num2str(n) '/' num2str(length(nets_trial))]);
    a = ExtractDataAtLayer(nets_trial{n}, shapes,targets,L);
    if(L==2), a = compet(a); end
    C = corr(a); C = bsxfun(@max,C,C');

%     Keep eigenvector e of correlation matrix
    [V,D] = eig(C);
    d = diag(D);
    [d,ind] = sort(d, 'descend'); %sort in descending order
    V = V(:,ind);
    Vn(:,:,n) = V;
end

% fix eigenvector, & pairwise max-correlate the eigenvectors across nets
display('max-correlate the eigenvectors across nets');
for e = 1:length(d); % eigenvector to compare
    display(['Calculating Pair-wise eigenvector correlations across NNs for eigenvector ' num2str(e)]);
    Ve = squeeze(Vn(:,e,:));
    Cv = zeros(size(Ve,2)); % numNets x numNets
    
    rho1 = corr(Ve);
    rho2 = corr(Ve,-Ve);
    Cv = max(rho1,rho2);
    EigCorrs{e} = Cv;
end

% plot average/std of this pairwise matrix for each eigenvector
display('plotting avg of max-corr matrix vs eigenindex');
mns = [];
v = [];
for i = 1:length(EigCorrs)
    mns =[mns mean(mean(EigCorrs{i}))];
    v = [v var(EigCorrs{i}(:))];
end
figure, errorbar(1:length(EigCorrs), mns, v); 
title('Average Activation-correlation pairwise eigenvector max-correlation across NNs')
xlabel('Eigenvector index');
ylabel('Mean Correlation');
makeFiguresPretty;


%% Answer Question: How small of 1 HL do you need to get good error?
numTrials = 30; % average the error per identical architecture over this many trials
datafile = './Data/DigitalManifold-30.mat';
layers = [1 5 10 20:10:100];
opts.max_fail=200;
% parpool('local', 3);

% Train the NNs
nets = {};
for i = 1:length(layers)
    display(['Training Net ' num2str(i) '/' num2str(length(layers)), ' size =' num2str(layers(i)) ]);
    for j = 1:numTrials
        opts.layers = layers(i);
        [netT] = TrainNN(datafile, opts);
        nets{i,j} = netT;
        display(['   Trial ' num2str(j) ' GenError = ' num2str(nets{i,j}.userdata.genError)]);
    end
    if(mod(i,5) == 0)
        save([num2str(i) ' Trained 0HL NN over ' num2str(j) ' Trials'], 'nets')
    end
end

    %% Test the nets and plot their average error as a function of HL size
    load('./Data/DigitalManifold-30.mat')    
    err = zeros(size(nets));  
    for i = 1:size(nets,1)
        for j = 1:size(nets,2)
            err(i,j) = nets{i,j}.userdata.genError;
%             err(i,j) = TestNN(nets{i,j}, shapes, targets);
        end
    end
    s = var(err, 0, 2);
    merr = mean(err,2);    
    figure, errorbar(layers, merr, s, '-r');
    title(['Error vs HL Size (1HL): ' num2str(numTrials) ' Trials per layer']);
    xlabel('# Hidden Units');
    ylabel('Generalization Error - 100 pts');
    
    % Plot all numTrials plots
    figure, hold on;
    for i =1:size(err,2)
        plot(layers, err(:,i));
    end
    
    makeFiguresPretty;

%% Plot layer by layer dim reduction
% given neT, shapes, targets
[~, t] = max(targets);
for i = 1:netT.numLayers
    data = ExtractDataAtLayer(netT, shapes, targets, i);
    ApplyDimReduction(data', t, 'pca');
    
end

initnw
%% Check classification of Random Matrices.... - classification is worthless
load('./Data/DigitalManifold-30'); % shapes, targets

% normalize data
shapes = shapes * 2 - 1;

numTrials = 500;
errs = zeros(numTrials,1);
[~, t] = max(targets,[],1);
for i = 1:numTrials
    weights = rand(3,900)* 2 - 1;
    bias = rand(3,1) * 2 - 1;
    out = weights * shapes + repmat(bias, 1, size(shapes,2));
    [~,labels] = max(out,[],1);
    e = sum(labels ~= t) / length(labels)
    errs(i) = e;
end
mean(errs)
%% Plot Weight Matrices for DNN NN
w = nn.W{1};

% Find good subplot dimensions to use
N = size(w,1);
M = 2;
p = perms(factor(N)) ;
if length(p) >= M,
  y = unique([p(:,1:M-1) prod(p(:,M:end),2)],'rows') ;
else
  error('No solution')
end
dif = max(y,[],2) - min(y,[],2);
[~,idx]=min(dif);
splot_ind = y(idx,:);

% Plot
for i =1:N
    subplot(splot_ind(1), splot_ind(2), i);
    imagesc(reshape(w(i,1:end-1), 28,28));
    colormap('gray');
%     colorbar;
end

%% Subhy/Surya - plot neuron outputs for all shapes
% load('./Data/DigitalManifold-30.mat'); % Shapes, targets - 1200 images

[~, t] = max(targets); 

figure; colorbar;
idx = 1;
for i = 1:3
    for j = 1:3
        shape1 = shapes(:, t == i);
        output = netT.IW{1} * shape1 + repmat(netT.b{1},1,size(shape1,2));
        subplot(3,3,idx), imagesc( reshape(output(j,:),40,40) ); title(['shape: ' num2str(i) ' output: ' num2str(j)]); colorbar;
        idx = idx + 1;
    end
end

%%

% Weights
for i = 1:3
    iw = netT.IW{1};
    figure; imagesc( reshape( iw(1,:), 60,60 ) ) ;
end

% try random weights as 
% trainine error to 0  & create same neuronalOutput(shape) plots from cell
% above
% calculate neuron weight-weighted average images (i.e. neuron features)
% look at autoencoders - use DNN toolbox

%% Pairwise Distance Matrices
load('./Data/Trained on digital/Net2/TNet-2HL-40 20-TrSetSize=1100-Epoch=20000.mat'); % netT, test_x, test_y, train_x, train_y
load('./Data/DigitalManifold-30.mat'); % shapes, targets

% X = shapes; % Raw
X = ExtractDataAtLayer(netT, shapes, targets, 1);
X = ExtractDataAtLayer(netT, shapes, targets, 2);
% X = ExtractDataAtLayer(netT, shapes, targets, 3);

D = pdist(X');
M = squareform(D); % M(i,j) is distance between ith and jth objects in original data
figure;
imagesc(M);

title('Pairwise Distance Matrices: [Rect, Tri, Circ], DM30, Layer 3 Output');
colorbar;
makeFiguresPretty

%% Train Many identical NN and create correlation matrices of their weights
% When I first did this (May 7th), it yielded nets with very low training
% error (or so I thought). Later, when I checked the nets, they had very
% high errors. It could be possible that my display was screwing up an
% order of magnitude, and what I thought was 2.18% error was actually 21.8%
load('./Data/AnalogManifold-30.mat'); %shapes, targets
numNN = 200;
opts.layers = 20;

% Train the NNs
nets = {};
for i = 1:numNN
    display(['Training Net ' num2str(i)]);
    [netT] = TrainNN(shapes, targets, opts);
    nets{end+1} = netT;
    if(mod(i,25) == 0)
        save([num2str(i) ' Trained 1HL NN'], 'nets')
    end
end

    % Save Nets
    save([num2str(length(nets)) ' Trained 1HL NN - AnalogManifold'], 'nets');

    %% Compile Net files
    n = [5 30 98];    
    nnets = {};
    for i = n;
        load([num2str(i) ' Trained 0HL NN.mat'])
        nnets = {nnets{:} nets{:}}';
    end

    %% Test the nets
    err = [];
    for i = 1:length(nets)
        err = [err 100*TestNN(nets{i}, shapes, targets)]; %#ok<AGROW>
        display(['Net ' num2str(i) ': Average Error=' num2str(err(i)) '%']);        
    end

    %% Keep Nets thats are good enough
    errThresh = 0; % keep nets with less than errThresh average error
    nets0 = nets(errThresh >= err);

    %% Correlation matrix C_{mu,eta}^L for fixed mu, eta varies over (x,y,shape), L = layer
    L = 1;
%     mu = 210; Mu='square'; % centered square
%     mu = 610; Mu='triangle'; % centered triangle
    mu = 1010; Mu='circle'; % centered circle
    eta = 1:400; Eta='square';
%     eta = 401:800; Eta='triangle';c
%     eta = 801:1200; Eta='circle';
    
    a = ExtractDataAtLayer(nets0{2}, shapes, targets, L); %layer = 0 is input, 1 is HL, 2 is output
    if(L==2), a = compet(a); end
    s = a(:,mu);
    C = corr(s,a(:,eta));
    C = reshape(C, 20,20);
    figure, imagesc(C);
    colorbar;
    title(['Correlation matrix C_{mu,eta}^L for fixed mu=' Mu ', eta varies over (x,y,' Eta '), L = layer' num2str(L)]);
    makeFiguresPretty;
    
    %% Eigenvectors of the whole matrix of correlations
    n = 2;
    for L = 1:1
        a = ExtractDataAtLayer(nets0{n}, shapes, targets, L); %layer = 0 is input, 1 is HL, 2 is output
        if(L==2), a = compet(a); end
        C = corr(a);                  
        C = bsxfun(@max, C, C');% Enforce C is symmetric (its numerically slightly off, which is yielding complex eigenvalues/vectors)
        
        % Eigenvalues/Vectors
        [V,D] = eig(C);
        d = diag(D);
        [d,ind] = sort(d, 'descend'); %sort in descending order
        V = V(:,ind);
        
        % Plots of Eigenvalues
        for e = 1:3;
%         for e = length(d)-2:length(d)
            figure, imagesc(reshape(V(:,e), 20,60));
            colorbar;
%             title(['Eigenvector ' num2str(e) ' of ' SH1 '-' SH2 ' correlation matrix at layer ' num2str(L)]);
            title(['Eigenvector ' num2str(e) ' (lambda=' num2str(d(e)) ')  of layer ' num2str(L)]);
        end
        
        % PCA on the correlation matrix
        Y = [ones(1,400), 2*ones(1,400), 3*ones(1,400)];
        ApplyDimReduction(C,Y,'pca');
    end
    
    %% PCA on vectorized low-rank approximation to correlation matrix C_{mu,eta}^{l,n}: mu,eta index shape
    L = 1;
    k = 3;
    M = size(shapes,2); % 
    t = zeros(length(nets0), M^2);
    for i = 1:length(nets0)
        % Low Rank Approximation to Correlation Matrix
        a = ExtractDataAtLayer(nets0{i}, shapes, targets, L); %layer = 0 is input, 1 is HL, 2 is output
        if(L==2), a = compet(a); end
        C = corr(a);                  
        C = bsxfun(@max, C, C');% Enforce C is symmetric (its numerically slightly off, which is yielding complex eigenvalues/vectors)
        Ck = LowRankApproximation(C,k);
        t(i,:) = Ck(:);        
    end
    ApplyDimReduction(t,[],'pca');
    
    
    %% Gather Eigenvectors of the correlation matrix across neural nets
    L = 1; % layer    
    Vn = zeros(length(nets0{1}),length(nets0{1}), length(nets0));
    for n = 1:length(nets0)
        n
        a = ExtractDataAtLayer(nets0{n}, shapes,targets,L);
        if(L==2), a = compet(a); end
        C = corr(a); C = bsxfun(@max,C,C');
        
        % Keep eigenvector e of correlation matrix
        [V,D] = eig(C);
        d = diag(D);
        [d,ind] = sort(d, 'descend'); %sort in descending order
        V = V(:,ind);
        Vn(:,:,n) = V;
    end
        %% Compare fixed eigenvectors of the activation-correlation matrix across neural nets
    %     acpwevcNN = {}; %Activation-correlation pairwise eigenvector max-correlation across NNs
        tic;
        for e = 1:1200; % eigenvector to compare
            display(['Calculating Pair-wise eigenvector correlations across NNs for eigenvector ' num2str(e)]);
            Ve = squeeze(Vn(:,e,:));
%             Cv = zeros(size(Ve,2)); % numNets x numNets
%             for i = 1:size(Ve,2)
%                 for j = 1:size(Ve,2)
%                     Cv(i,j) = max( corr(Ve(:,i), Ve(:,j)) , corr(Ve(:,i), -Ve(:,j)) );
%                 end
%             end
            Cv = max( corr(Ve,Ve), corr(Ve,-Ve)); 
            Cv = max(Cv,Cv');
            
            acpwevcNN{e} = Cv;
%             toc;
    %         figure, imagesc(Cv); colorbar;
    %         title(['Correlation of eigenvector ' num2str(e) ' across ' num2str(length(nets0)) ' 1HL NNs']);
        end
            %% Average acpwevcnNN vs eigenvector index
            m = [];
            v = [];
            for i = 1:length(acpwevcNN)
                m = [m mean(mean(acpwevcNN{i}))];
                v = [v var(acpwevcNN{i}(:))];
            end
            figure, errorbar(1:length(acpwevcNN), m, v); 
            title('Average Activation-correlation pairwise eigenvector max-correlation across NNs')
            xlabel('Eigenvector index');
            ylabel('Mean Correlation');


        %% PCA on the first 100 eigenvectors      
        vect = @(v) v(:);
        acpmat = cell2mat( cellfun(vect, acpwevcNN, 'UniformOutput', 0) )';
        b = 8; Y = [ones(1,b), 2*ones(1, 100-b)];
        pca_results = ApplyDimReduction(acpmat, Y, 'pca');
    
    %% PCA on vectorized correlation matrices (correlating all activations) of the Lth layer
    L = 1; 
    cms = zeros(length(nets0), 1200^2);
    for i = 1:length(nets0)
        display(['Gathering vectorized correlation matrices of layer ' num2str(L) ', net ' num2str(i)]);
        a = ExtractDataAtLayer(nets0{i}, shapes, targets, 1); % nth hidden layer
        c =  corr(a);
        cms(i,:) = c(:)';
    end
    ApplyDimReduction(cms,[],'pca');
    
    %% Calculate activations at a particular layer 
%     activs = zeros(length(nets0) , numel(targets)); % Output activation
    activs = zeros(length(nets0) , numel(ExtractDataAtLayer(nets0{1}, shapes, targets, 1))); % 1st hidden layer
    for i = 1:length(nets0);
        display(['Calculating Activation ' num2str(i) '/' num2str(length(nets0))]);
%         a = nets0{i}(shapes); % Output activation
        a = ExtractDataAtLayer(nets0{i}, shapes, targets, 1); % nth hidden layer
        activs(i,:) = a(:);
    end
    
    %% PCA of activations
    ApplyDimReduction(activs, [], 'pca');
    
    %% Calculate Similarity Vectors of activations 
    sims = zeros( length(nets0), size(shapes, 2)^2 );
%     outputSize = nets0{1}.output.size; % output size
    outputSize = nets0{1}.layers{1}.size; % layer size
    for i = 1:length(nets0)
        display(['Calculating Similarity Vector ' num2str(i) '/' num2str(NN)]);
        a = reshape( activs(i,:), outputSize, []); % Grab the activations in outputSize x numExamples format
        sims(i,:) = reshape(a'*a, 1, []); % Calculate vectorized similarity matrix
    end
    
    %% Calculate correlation matrix of the similarity vectors
    rho = corr(sims');
    figure, imagesc(rho);
    title(['Corr Matrix: Similarity Vectors of ' num2str(length(nets0)) ' 1-HL NNs trained to ' num2str(errThresh) '% error']);
    colorbar;
    makeFiguresPretty;
    
    %% PCA on the similarity matrices
    ApplyDimReduction(sims, [], 'pca')
    
    %% Hierarchical Agglomerated Clustering of the correlation matrix of similarity vectors
    Z = linkage(rho, 'ward', 'euclidean');
    [H,T, OUTPERM] = dendrogram(Z,0);
    figure, imagesc( rho(OUTPERM, OUTPERM) ), colorbar;
%     c = OUTPERM;
    c = cluster(Z, 'maxclust', 3);
    
    % Shuffle Rows
    rho_clustered = rho(c == 1,:);
    for i = 2:max(c)
        rho_clustered = [ rho_clustered; rho(c == i, :)];
    end
    
    %Shuffle Columns
    rho_bd = rho_clustered(:, c == 1);
    for i =2:max(c)
        rho_bd = [ rho_bd, rho_clustered(:, c == i)];
    end
    
    figure, imagesc(rho_bd);
    title(['Clustered Corr Matrix: Similarity Vectors of ' num2str(length(nets0)) ' 1-HL NNs trained to ' num2str(errThresh) '% error']);
    colorbar;
    makeFiguresPretty;
    
    %% Correlation Matrix of Input Weights
    sIdx = 2; 
    net = netsG;
    W = zeros(size(net{1}.IW{1}, 2), length(net));
    for i = 1:length(net)    
        iw = net{i}.IW{1};
        W(:,i) = iw(sIdx,:)';
    end
    rho = corr(W);
    figure, imagesc(rho); 
    title(['Corr Matrix: 1-HL NNs Input Weights, trained on DigitalManifold-30, OutputNeuron=' num2str(sIdx)]);
    colorbar;
    
    
     %% Correlation Matrix of Layer Weights
    sIdx = 10; 
    layer = 1;
    W = zeros(size(nets0{1}.LW{layer+1, layer}, 2), length(nets0));
    for i = 1:length(nets0)            
        iw = nets0{i}.LW{layer+1,layer};
        W(:,i) = iw(sIdx,:)';
    end
    rho = corr(W);
    figure, imagesc(rho); 
    title(['Corr Matrix: 1-HL NNs layer ' num2str(layer) ', trained on DigitalManifold-30, OutputNeuron=' num2str(sIdx)]);
    colorbar;
    makeFiguresPretty;

%% 


%% SWEEP NUMBER OF UNITS IN A HIDDEN LAYER OF A DNN - GENERATE DATA ON THE FLY
for Wrapper = 1:1
    %    DEFINING THE SHAPE MANIFOLD
    imsize = 30;
    num_shapes = 1500; 
    [shapes, targets] = CreateManifold(imsize, num_shapes); 

    % SWEEP OF # OF UNITS IN THE HIDDEN LAYER
    networkErrors = [];
    max_test = 15;
    fixedHiddenLayers = [40];
    HLSizes = [20];

    for j = 1:length(HLSizes)
        shl = HLSizes(j);
        display(['Training network with [' num2str(fixedHiddenLayers) ' ' num2str(shl) '] units in the HLs']);

        %    DEFINING THE NETWORK
        layers = [fixedHiddenLayers shl];
        net = feedforwardnet(layers, 'traingdm');    
        net.trainParam.epochs = 10000;

        %    TRAINING THE NETWORK
        net.divideFcn = '';
        net.trainParam.goal = 0.001; % Mean squared error of 0.001 is performance metric
        [netT,tr] = train(net,shapes,targets);
        save(['TNet-' num2str(length(layers)) 'HL-' num2str(layers) '-TrSetSize=' num2str(size(shapes,2)) '-Epoch=' num2str(net.trainParam.epochs)], 'netT');

        % TESTING THE NETWORK
        display(['Testing network with ' num2str(shl) ' units in the HL']);

        errors = 0;
        for i = 1:max_test
            display(['   test iter ' num2str(i)]);
            [P, T] = CreateManifold(imsize, floor(num_shapes/10)); 

            % TEST NETWORK 1
            A = netT(P);
            AA = compet(A);
            errors = errors + sum(sum(abs(AA-T)))/2;
        end
        networkErrors = [networkErrors errors/(size(P,2) * max_test)];
    end
end


%% Plot image-averaged input weight matrices in 8x5 confi
load('./Data/Trained on digital/TNet-2HL-40  20-TrSetSize=1100-Epoch=10000.mat');
load('./Data/DigitalManifold-30.mat'); % shapes is 900x 1200

% shapes = shapes(:,1);

% Hidden Layer 1
figure;
iw = netT.IW{1}; %40 x 900
nlfun1 = str2func(netT.layers{1}.transferFcn);
neuron_outputs = nlfun1(iw*shapes); % 40 x 1200
image_weighted_avgs = (shapes * neuron_outputs') / size(shapes, 2);

for i = 1:size(image_weighted_avgs,2)
    subplot(8,5,i), imagesc(reshape(image_weighted_avgs(:,i), 30,30))
end
colormap('gray')
    
% Hidden Layer 2
figure;
nlfun2 = str2func(netT.layers{2}.transferFcn);
neuron_outputs = nlfun2( netT.LW{2,1} * neuron_outputs);
% neuron_outputs = tansig(netT.LW{2,1} * tansig(netT.IW{1} * shapes));
image_weighted_avgs = (shapes * neuron_outputs') / size(shapes, 2);

for i = 1:size(image_weighted_avgs,2)
    subplot(4,5,i), imagesc(reshape(image_weighted_avgs(:,i),30,30))
end
colormap('gray')

% Output Layer
figure;
% lw = netT.LW{3,2};
ofun3 = str2func(netT.layers{3}.transferFcn);
neuron_outputs = ofun3(netT.LW{3,2} * neuron_outputs);
% neuron_outputs = tansig(netT.LW{3,2} * tansig(netT.LW{2,1} * tansig(netT.IW{1} * shapes)));
image_weighted_avgs = (shapes * neuron_outputs') / size(shapes, 2);

for i = 1:size(image_weighted_avgs,2)
    subplot(1,3,i), imagesc(reshape(image_weighted_avgs(:,i),30,30))
end
colormap('gray')



%% SCAN THROUGH TRAINED NET DATA STRUCTURES
%  TEST THEM ON A NEWLY GENERATED MANIFOLD
%  OBTAIN NETWORK ERRORS
for Wrapper=1:1
    d = dir('./TNet-2HL*');
    d = {d.name};
    imsize = 30;
    num_shapes = 15000;
    networkErrors = [];
    max_test = 10;


    for i = 1:length(d)

        % LOAD NETWORK
        load(d{i});

        % TEST NETWORK
        display(['Testing Network ' d{i}]);
        errors = 0;
        for j = 1:max_test
            display(['   test iter ' num2str(j)]);
            [P, T] = CreateManifold(imsize, num_shapes); 

            % TEST NETWORK 1
            A = netT(P);
            AA = compet(A);
            errors = errors + sum(sum(abs(AA-T)))/2;
        end
        networkErrors = [networkErrors errors/(size(P,2) * max_test)];    
    end

    % %% PLOT ERRORS VS # UNITS IN HIDDEN LAYER
    % 
    % plot(HLSizes, networkErrors, '-')
    % title(['imsize=' num2str(imsize) '^2, ',...
    %        '# of each shape=' num2str(num_shapes_per_type) ', ',...   
    %        'HL=[100 N]',...
    %        ]);
    % xlabel('# of units in the HL');
    % ylabel('Error Fraction');
    %        
    % makeFiguresPretty;
    %    
    %    
end
