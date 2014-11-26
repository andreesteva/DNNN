function heatmaps = ViewNeuron_FunctionSpace(neuron, net, layer, stimuli, shape_names)
% function ViewNeuron_FunctionSpace(neuron, net, layer, all_shapes)
% plot neurons as functions on stimulus space for any chosen 
%neuron in any chosen network in any chosen layer (which we can
% choose in real time when we next meet)
%
% neuron is a whole number index in the range 1:N_layer where N_layer is
% the number of neurons in 'layer'
%
% net is trained matlab neural network
% 
% layer is a non-negative integer specifying the layer of consideration (0
% means the input layer, in which case this function calls
% ViewPixel_FunctionSpace)
%
% stimuli is a dim x num_images matrix where dim is the dimensionality
% of the image (tpyically 30x30=900)
%
% shape_names is the name of each of classes in the dataset, which is assumed to
% be sequential such that all images of the same class are contigious
% columns in the matrix all_shapes


    % Feedforward pass
    data = ExtractDataAtLayer(net, stimuli, layer);
    num_shapes = length(shape_names);

    % figure out number of images and number of images per shape-class
    num_images = size(data,2);
    images_per_shape = num_images / num_shapes;
    
    % Plot heatmaps
    heatmaps = cell(1, num_shapes);   
    for i = 1:num_shapes
        images = data(:,(i-1)*images_per_shape + 1 : i*images_per_shape);
        func = reshape(images(neuron,:), sqrt(images_per_shape), sqrt(images_per_shape));
        figure, imagesc(func); colorbar;
        title(shape_names{i});
        heatmaps{i} = func;
    end

end


