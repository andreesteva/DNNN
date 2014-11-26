% function im = DrawPentagon(xc, yc, w, M)

% Draw square
[X,Y] = meshgrid(1:M, 1:M);
p1 = dualsigmoid(X-xc,w/2); p1 = 1 - p1;
p2 = dualsigmoid(Y-yc,w/2); p2 = 1 - p2;
f = p1 .* p2;

% Calculate 5 corner points for convex hull function to use


% Create 5 planes
plane1 = -min(3*(X - xc-w/2) + (Y - yc)-0.75*w, 0);
plane2 = -min(-3*(X - xc+w/2) + (Y - yc)-0.75*w, 0);

f = f .* plane1 .* plane2;
imagesc(f)