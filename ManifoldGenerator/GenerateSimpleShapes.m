function [shapes] = GenerateSimpleShapes(IMSIZE, PTS, VRNTS, NUM_SHAPES, SHAPE, FILL, BG_COLOR, NAME, MAX_STEP_SIZE, FILTER)
% Shapes is a IMSIZE^2 X NUM_SHAPES matrix where each column represents
% the concatenated columns of an image (that is, it each column =
% reshape(image, numel(image),1)
% 
% the shapes are jittered - they are translated, rotated, shrunk, tilted,
% and reshaped

% figure;
pos = PTS;
shapes = zeros(IMSIZE^2, NUM_SHAPES);
shapeInserter = vision.ShapeInserter('Shape', SHAPE, 'Fill', FILL);
for i =1:NUM_SHAPES
    % Create a gray background
    im_gray = ones(IMSIZE) * BG_COLOR;

    % Put a shape on it
    im_shape = step(shapeInserter, im_gray, pos);
    
    % Blur the image
    h = FILTER;
    im_shape = imfilter(im_shape, h, BG_COLOR);

    % Display and save as jpg
%     im = imshow(im_shape);
%     saveas(im, [NAME num2str(i)], 'jpg');   
    shapes(:,i) = reshape(im_shape, numel(im_shape), 1);

    % Position of next shape
    pos = pos + (2*rand(size(VRNTS))-1).*VRNTS*MAX_STEP_SIZE; 
    if( any(pos(pos > IMSIZE)) || any(pos(pos <= 0)) )
        pos = PTS; % reset to original if it goes out of bounds
    end
%     pos(pos > IMSIZE) = IMSIZE;
%     pos(pos <= 0) = PTS(pos <= 0);
end