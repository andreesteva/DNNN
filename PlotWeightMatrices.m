function PlotWeightMatrices(nn, type)
    % nn is a neural network
    % type is either 'DNN' or 'MNN'
    % DNN means its from the github NN toolbox
    % MNN means its from the Matlab NN toolbox

    if(strcmpi(type, 'dnn')) % GITHUB NN
        for j = 1:length(nn.W)
            w = nn.W{j};
            w = w(:,1:end-1);
            PlotWeightMatrix(w);
%             title(['Layer ' num2str(j)]);
        end
    elseif(strcmpi(type, 'mnn')) %MATLAB NN
        for j = 0:numel(nn.LW)
            if(j == 0)
                w = nn.IW{1};
            else
                w = nn.LW{j};
            end
            if(isempty(w)) continue; end            
            PlotWeightMatrix(w);
%             title(['Layer ' num2str(j)]);

        end
    end

end
