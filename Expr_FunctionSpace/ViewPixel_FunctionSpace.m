function ViewPixel_FunctionSpace(pixels, all_shapes, num_shapes)
% function ViewPixelHeatMaps(pixel, shapes)
% 
% pixels is [x y] pixel coordinates
% all_shapes is the set of all shapes in the manifoldin the matrix format
% [dimensionality x number_of_instances], and is assumed to contain square
% images
% the sequence of shapes in all_shapes is assumed to be first in y then in
% x (i.e. the shapes translate first down then across in the image)
% num_shapes is typically 3, for triangle, circle, and square, but can be
% more, for more complex manifolds

% Check valid pixel
if(prod(pixels) > size(all_shapes,1))
    error('pixel out of range');
end

% Image dimensionality
image_dimension = size(all_shapes,1);
dim = sqrt(image_dimension);

% figure out number of images and number of images per shape-class
num_images = size(all_shapes,2);
images_per_shape = num_images / num_shapes;

% Plot heatmaps
for i = 1:num_shapes
    images = all_shapes(:,(i-1)*images_per_shape + 1 : i*images_per_shape);
    pixel_ind = pixels(2) + (pixels(1)-1)*dim;
    func = reshape(images(pixel_ind,:), sqrt(images_per_shape), sqrt(images_per_shape));
%     HeatMap(func);
    figure, imagesc(func); colorbar;
end

