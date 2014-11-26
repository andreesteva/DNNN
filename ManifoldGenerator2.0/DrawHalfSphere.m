function im = DrawHalfSphere(xc,yc,r,M)

% Paraboloid
[X,Y] = meshgrid(1:M, 1:M);
f = min(((X-xc).^2 + (Y-yc).^2)/r^2, 1);

% Invert - set BG to black
f = 1-f;

% Contruct half-Plane
plane = -min(-sigmoid(Y - yc), 0);

% Create half-sphere from circle and half-plane
f = f .* sigmoid(Y-yc);

% Return
im = f;