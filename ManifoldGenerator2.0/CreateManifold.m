% Create Manifold

% xc = 15.1;
% yc = 15.1;
% r = 5;
% M = 30;
% 
% im = DrawCircle(xc, yc, r, M);
% figure, imagesc(im);
% colormap('gray');
% 
% im = DrawSquare(xc, yc, r*2, M);
% figure, imagesc(im);
% colormap('gray');
% 
% im = DrawTriangle(xc, yc, r*2, M);
% figure, imagesc(im);
% colormap('gray');

tic;
% Define manifold
M = 30;
r = 5; % radius of circle, half-width of square and triangle
dimrange = 153; % do 153 for a 70k dataset


% Settings
xmin = r;
xmax = M-r;
xc = xmin:(xmax-xmin)/(dimrange-1):xmax;
yc = xc;
[X,Y] = meshgrid(xc,yc);

% Generate Circles
drawcirc = @(x,y) DrawCircle(x,y,r,M);
circles = arrayfun(drawcirc, X,Y, 'UniformOutput', false);
toc

% Generate Squares
drawsquare = @(x,y) DrawSquare(x,y,2*r,M);
squares = arrayfun(drawsquare, X,Y, 'UniformOutput', false);
toc

% Generate Triangles
drawtriangle = @(x,y) DrawTriangle(x,y,2*r,M);
triangles = arrayfun(drawtriangle, X,Y, 'UniformOutput', false);
toc

% % Generate Rectangle
% drawrectangle = @(x,y) DrawRectangle(x,y,2*r,M);
% rectangles = arrayfun(drawrectangle, X,Y, 'UniformOutput', false);
% toc
% 
% % Generate Half-Spheres
% drawhalfsphere = @(x,y) DrawHalfSphere(x,y,r,M);
% halfspheres = arrayfun(drawhalfsphere, X,Y, 'UniformOutput', false);
% toc

% Reshape into 900x1 vectors
straighten = @(mat) mat(:);
circles = cellfun(straighten, circles, 'UniformOutput', false); 
squares = cellfun(straighten, squares, 'UniformOutput', false); 
triangles = cellfun(straighten, triangles, 'UniformOutput', false); 
% rectangles = cellfun(straighten, rectangles, 'UniformOutput', false);
% halfspheres = cellfun(straighten, halfspheres, 'UniformOutput', false);

% Create shapes matrix
% shapes = cell2mat([circles(:); squares(:); triangles(:); rectangles(:); halfspheres(:)]');
shapes = cell2mat([circles(:); squares(:); triangles(:)]');
toc

% Create targets matrix
% %rtc goes 1 2 3
% s = [1 0 0 0 0]';
% t = [0 1 0 0 0]';
% c = [0 0 1 0 0]';
% r = [0 0 0 1 0]';
% hs= [0 0 0 0 1]';
% targets = [repmat(s, 1, dimrange^2), repmat(t, 1, dimrange^2), repmat(c, 1, dimrange^2), repmat(r, 1, dimrange^2), repmat(hs,1,dimrange^2)];

s = [1 0 0]';
t = [0 1 0]';
c = [0 0 1]';
targets = [repmat(s, 1, dimrange^2), repmat(t, 1, dimrange^2), repmat(c, 1, dimrange^2)];

toc

%% Observe
ind = randperm(length(shapes));
figure;
for i = 1:30
    imagesc(reshape(shapes(:,ind(i)),M,M))
    pause(0.3)
end

%% save
save('ThreeShapeManifold-30', 'shapes', 'targets');

%%
figure, imagesc( reshape(circles{4},30,30));
colormap('gray');

%%
figure, imagesc( reshape(shapes(:,70000),30,30));
colormap('gray');