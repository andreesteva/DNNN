%%  Creates a Digital Manifold - 
% objects are centered/located at pixel centers

% Parameter Defs
imsize = 60; % imsize x imsize image
shapesize = 20; % shapes will fill a shapesize x shapesize ROI
filter = fspecial('gaussian', round([imsize imsize]/10), round(imsize/5));
imagesFolder = './Manifold Shapes/'; % puts the jpgs in here
nspc = (imsize - shapesize)^2;
datasetsize = nspc * 3;
shapes = [];
targets = [];
BGcolor = 0.5;

% Create CV Objects
im_gray = ones(imsize) * BGcolor;
si_rect = vision.ShapeInserter('Shape', 'Rectangles', 'Fill', true); % CV objects to insert the shapes into the images
si_tri = vision.ShapeInserter('Shape', 'Polygons', 'Fill', true);
si_circ = vision.ShapeInserter('Shape', 'Circles', 'Fill', true);

% Generate shapes on the gray image
for i = 0:(imsize-shapesize) - 1
    for j = 0:(imsize-shapesize) - 1
        
        % Rectangle - place top left corner of rectangle in (i,j)
        pos_rect =[i j shapesize shapesize];
        shapes(:,end+1) = reshape( imfilter(step(si_rect, im_gray, pos_rect), filter, BGcolor), imsize^2, 1);
        targets(:,end+1) = [1 0 0]';
    end
end
for i = 0:(imsize-shapesize) - 1
    for j = 0:(imsize-shapesize) - 1 
        % Triangle - add i to the first dimension of the image and j to the second (the first dimension goes up and down)
        pos_tri = [i j i j+shapesize i+shapesize j];
        shapes(:,end+1) = reshape( imfilter(step(si_tri, im_gray, pos_tri), filter, BGcolor), imsize^2, 1);
        targets(:,end+1) = [0 1 0]';
    end
end
for i = 0:(imsize-shapesize) - 1
    for j = 0:(imsize-shapesize) - 1
        % Circle - place center in (i+shapesize/2, j+shapesize/2) and make its radius shapesize/2)
        pos_circ = [i+shapesize/2 j + shapesize/2 shapesize/2];
        shapes(:,end+1) = reshape( imfilter(step(si_circ, im_gray, pos_circ), filter, BGcolor), imsize^2, 1);
        targets(:, end+1) = [0 0 1]';
    end
end

%% For Saving

save(['DigitalManifold-' num2str(imsize)], 'shapes', 'targets');

%% For Displaying
figure;
for i = 1:size(shapes,2)
    imagesc(reshape(shapes(:,i),imsize,imsize));
    pause(0.01);
end