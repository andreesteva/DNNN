function im = DrawRectangle(xc, yc, w, M)
% function im = DrawRectangle((xc, yc, w, M);
% Returns an MxM matrix im of all zeros with an upright rectangle drawn 
% the fixed aspect ratio is supposed to be 4:1, the sigmoidal blurring
% fudge this a bit
% M is size of image in pixels
% w is the height of the rectangle
% (xc, yc) is center of the rectangle

% Settings
% M = 30;
% w = 10; 
% xc = 5.2;
% yc = 5.2;


[X,Y] = meshgrid(1:M, 1:M);
p1 = dualsigmoid(X-xc,w/8); p1 = 1 - p1;
p2 = dualsigmoid(Y-yc,w/2); p2 = 1 - p2;

f = p1 .* p2;

% Return image
im = f;