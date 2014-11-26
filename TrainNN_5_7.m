function [netT, tr] = TrainNN_5_7(dataFile, cutoff, layers)
% Old function - used with the Drawing Board to train a ton of NN in a loop

    load(dataFile); % Shapes, targets - 1200 images

    % Choose testing/training data
    ind = randperm(size(shapes,2));
%     cutoff = cutoff;
    train_ind = ind(1:cutoff);
    test_ind = ind(cutoff+1:end);
    train_x = shapes(:,train_ind); train_y = targets(:, train_ind);
    test_x = shapes(:, test_ind); test_y = targets(:, test_ind);

    % Set up & Train NN - this is the setup I used on Wednesday May 7th for
    % the plots I created. 
%     layers = [];
    net = feedforwardnet(layers, 'traingdm'); % Change to cross entropy objective with newer matlab version
    net.performFcn = 'mse'; % use 'mse' or 'crossentropy'
    net.trainParam.epochs = 100000;
    net.divideFcn = '';
    net.trainParam.goal = 0.001; % Mean squared error of 0.001 is performance metric
    display(['Training NN with layers = ' num2str(layers)]);
    [netT,tr] = train(net,train_x, train_y);
%     display(['Generalization Error: ' num2str(TestNN(netT, test_x, test_y)) '%']);
    display(['Generalization Error: ' num2str(TestNN(netT, test_x, test_y) * 100) '%']); % May 14th, 2014 - the *100
