function im = DrawTriangle(xc, yc, w, M)
% function im = DrawTriangle
% Returns an MxM matrix im of all zeros with an upper triangle drawn 
% M is size of image in pixels
% w is width of triangle
% (xc, yc) is center of square

% Settings
% M = 30;
% w = 10; 
% xc = 15.2;
% yc = 15.6;

% Empty Grid
im = ones(M);

% Draw square
[X,Y] = meshgrid(1:M, 1:M);
p1 = dualsigmoid(X-xc,w/2); p1 = 1 - p1;
p2 = dualsigmoid(Y-yc,w/2); p2 = 1 - p2;
f = p1 .* p2;

% Create a semi plane
plane = -min((X - xc) + (Y - yc), 0);

% Create triangle from square and semi-plane
f = f .* plane;

% Normalize
f = f ./ max(f(:));

% return im
im = f;

% figure;
% imagesc(f);
% colormap('gray');o