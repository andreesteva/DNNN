function im = DrawCircle(xc, yc, r, M)
% function im = DrawCircle(xc, yc, l, M, R)
% Returns an MxM matrix im of all zeros with a parabola circle drawn 
% M is size of image in pixels
% r is radius of circle
% (xc, yc) is center of circle

% Settings
% M = 30;
% R = 10; % width of a square ROI surrounding the shape
% xc = 25.2;
% yc = 25.2;
% l = 5;


% Empty Grid
im = ones(M);

% Paraboloid
[X,Y] = meshgrid(1:M, 1:M);
f = min(((X-xc).^2 + (Y-yc).^2)/r^2, 1);
% f = dualsigmoid((X-xc).^2 + (Y-yc).^2, 2*r);

% Invert - set BG to black
f = 1-f;

% Zero values outside the ROI
% r = R/2;
% ROIx = round([1:R] + xc) - round(R/2);
% ROIy = round([1:R] + yc) - round(R/2);
% mask = zeros(M);
% mask(ROIx, ROIy) = 1;
% f = f .* mask;

% Return im
im = f;

% figure, imagesc(f)