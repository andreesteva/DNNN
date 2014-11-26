function data = ExtractDataAtLayer(netT, x, layer)
% function data = ExtractDataAtLayer(netT, x, y, layer)
%
% netT is a trained neural network
% x is data in the format features x data points
% y is only used for post-processing x in the event that layer ==
% length(netT)
% layer is an int, its the layer at which we'll extract 
% the data (0 returns the input data, 1 returns the data from the first hidden layer)
%
% note: this function applies netT's pre-processing function to x before
% data flow through the network, but only applies the post-processing
% function if we want to extract data at the last layer
%
% returns the value data in inverse ML format: features x data points

% Check layer == 0
if(layer == 0)
    data = x;
    return;
end

% % Pre-Process
% if(~isempty(netT.inputs{1}.processFcns))
%     if(strcmp(netT.inputs{1}.processFcns, 'mapminmax'))
%         [x, xs] = mapminmax(x);    
%         [~, ts] = mapminmax(y);
%     end    % %%Placeholder for future pre-processing functions%%
% end

% Pre-Process
if(~isempty(netT.inputs{1}.processSettings))
    x = mapminmax('apply', x, netT.inputs{1}.processSettings{1});
end

% Input Layer
f = str2func(netT.layers{1}.transferFcn);
x = f(netT.IW{1} * x + repmat(netT.b{1}, 1, size(x,2)));

if(layer == 1)
    data = x;
    return;
end
  

% Hidden Layers
for i = 2:length(netT.layers)-1
    f = str2func(netT.layers{i}.transferFcn);
    x = f(netT.LW{i,i-1} * x + repmat(netT.b{i},1, size(x,2)));
    
    if(layer == i)
        data = x;
        return;
    end
end

% Output Layer
f = str2func(netT.layers{end}.transferFcn);
x = f(netT.LW{end,end-1} * x + repmat(netT.b{end},1, size(x,2)));

% Post Process
if(~isempty(netT.outputs{end}.processSettings))
    x = mapminmax('reverse',x,netT.outputs{end}.processSettings{1});
end


% % Post-Process - only if we're at the final layer
% if(~isempty(netT.outputs{end}.processFcns) && isequal(layer, length(netT.layers)))
%     if(strcmp(netT.inputs{1}.processFcns, 'mapminmax'))
%         x = mapminmax('reverse', x, ts); 
%     end % %%Placeholder for future post-processing functions%%
% end

data = x;
