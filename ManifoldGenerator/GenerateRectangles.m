function GenerateRectangles(num_rects)
% Draws num_rects images of a dark gray rectangle translated around and reshaped on a light gray background 

% USER: Set Image properties
IMSIZE = 30;
SHAPE = 'Rectangles';
PTS = IMSIZE * [0.31 0.3 0.3 0.3];
NAME = 'Rect';
FILL = true;
NUM_SHAPES = num_rects;
MAX_STEP_SIZE = IMSIZE * 0.1;
BG_COLOR = 0.5;

% Create and save NUM_RECTS images
pos = PTS;
for i =1:NUM_SHAPES
    % Create a gray background
    im_gray = ones(IMSIZE) * BG_COLOR;

    % Put a shape on it
    shapeInserter = vision.ShapeInserter('Shape', SHAPE, 'Fill', FILL);
    im_shape = step(shapeInserter, im_gray, pos);
    
    % Blur the imag
%     h = ones(5,5)/25; % simple blurring filter
    h = fspecial('gaussian', [3 3], 5);
    im_shape = imfilter(im_shape, h, BG_COLOR);

    % Display and save as jpg
%     figure;
    im = imshow(im_shape);
    saveas(im, [NAME num2str(i)], 'jpg');   

    % Position of next rectangle
    pos = pos + (2*rand(size(pos))-1)*MAX_STEP_SIZE; 
    pos(pos > IMSIZE) = IMSIZE;
    pos(pos <= 0) = PTS(pos <= 0);
end