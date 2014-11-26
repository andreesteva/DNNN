function [shapes, targets] = CreateManifold(IMSIZE, NUM_SHAPES)
% Creates a total of N=3*floor(NUM_SHAPES/3) images of size IMSIZE x IMSIZE
% these images are equally distributed among rectangles, triangles, and
% circles
% shapes is a matrix of size IMSIZE^2 x N
% targets is a matrix of size 3 x N

% Parameters
% IMSIZE = 20; % N x N image sizes
NUM_SHAPES = floor(NUM_SHAPES/3);
FILTER = fspecial('gaussian', round([IMSIZE IMSIZE]/10), round(IMSIZE/5));
target_rect = [1 0 0]';
target_tri = [0 1 0]';
target_circ = [0 0 1]';
targets = [repmat(target_rect, 1, NUM_SHAPES) ,...
           repmat(target_tri, 1, NUM_SHAPES) ,...
           repmat(target_circ, 1, NUM_SHAPES)];
folder = './Manifold Shapes/';

% Generate Rectangles
SHAPE = 'Rectangles';
PTS = IMSIZE * [0.31 0.3 0.3 0.3];
VRNTS = [1 1 0 0]; % Allows these dimensions of PTS to vary - currently set to translation
NAME = [folder 'Rect'];
FILL = true;
MAX_STEP_SIZE = IMSIZE * 0.1;
BG_COLOR = 0.5;

rects = GenerateSimpleShapes(IMSIZE, PTS, VRNTS, NUM_SHAPES, SHAPE, FILL, BG_COLOR, NAME, MAX_STEP_SIZE, FILTER);

% Generate Triangles
SHAPE = 'Polygons';
NAME = [folder 'Triangle'];
PTS = IMSIZE * [0.5 0.3 0.5 0.6 0.25 0.5]; % Draws a triangle
VRNTS = 1; % translation
FILL = true;
MAX_STEP_SIZE = IMSIZE * 0.1;
BG_COLOR = 0.5;

polys = GenerateSimpleShapes(IMSIZE, PTS, VRNTS, NUM_SHAPES, SHAPE, FILL, BG_COLOR, NAME, MAX_STEP_SIZE, FILTER);

% Generate Circles
SHAPE = 'Circles';
NAME = [folder 'Circle'];
PTS = IMSIZE * [0.5 0.5 1/6]; %centered circle
VRNTS = [1 1 0 ]; % translation
FILL = true;
MAX_STEP_SIZE = IMSIZE * 0.1;
BG_COLOR = 0.5;

circles = GenerateSimpleShapes(IMSIZE, PTS, VRNTS, NUM_SHAPES, SHAPE, FILL, BG_COLOR, NAME, MAX_STEP_SIZE, FILTER);

shapes = [rects polys circles];