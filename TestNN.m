function err = TestNN(netT, X, Y)
% function err = TestNN(netT, X, Y)
% this function is the following line:
% err = sum(sum(abs(compet(netT(X)) - Y)))/2 /size(X,2); % compet does winner takes all on the output of netT
err = sum(sum(abs(compet(netT(X)) - Y)))/2 /size(X,2); % compet does winner takes all on the output of netT