function im = DrawSquare(xc, yc, w, M)
% function im = DrawSquare((xc, yc, w, M);
% Returns an MxM matrix im of all zeros with a square drawn 
% M is size of image in pixels
% w is width of square
% (xc, yc) is center of square

% Settings
% M = 30;
% w = 10; 
% xc = 5.2;
% yc = 5.2;

% Empty Grid
im = ones(M);

% Draw square
[X,Y] = meshgrid(1:M, 1:M);
 
% Paraboloid square
% p1 = min(((X-xc).^2)/(w/2)^2, 1); p1 = 1 - p1;
% p2 = min(((Y-yc).^2)/(w/2)^2, 1); p2 = 1 - p2;

% sigmoidal square
p1 = dualsigmoid(X-xc,w/2); p1 = 1 - p1;
p2 = dualsigmoid(Y-yc,w/2); p2 = 1 - p2;

f = p1 .* p2;

% Return im
im = f;

