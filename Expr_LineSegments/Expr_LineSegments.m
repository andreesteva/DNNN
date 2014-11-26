% This experiment file will generate lines in image space that connect a
% triangle, a square, and a circle together, and pass them through a few
% trained nets, to observe the net's effect on the tendrils. These three
% shapes must be overlapping.

%% Step 1: Load three overlapping shapes from the analog manifold
load('/Users/AndreEsteva/Google Drive/Documents/Stanford/Stanford Vision Lab/DNNN/Data/AnalogManifold-30.mat') % shapes, targets

n=76*153 + 76; % index to a shape in the center
L = 70227/3; % index translation to get to different shape
circle_idx = n;
square_idx = n + L;
triangle_idx = n + 2*L;

circle = shapes(:,circle_idx);
square = shapes(:, square_idx);
triangle = shapes(:, triangle_idx);

%% Step 2: Create a sequence of points, arbitrarily close, that connect these three shapes
num_pts = 100;

% Circle-Square
tv_cs = (circle - square) * [0:num_pts]./num_pts;   % translation vectors
pts_cs = gadd(square, tv_cs);                       % line segment

% Square-Triangle
tv_st = (square - triangle) * [0:num_pts]./num_pts;   % translation vectors
pts_st = gadd(triangle, tv_st);                       % line segment

% Triangle-Circle
tv_tc = (triangle - circle) * [0:num_pts]./num_pts;   % translation vectors
pts_tc = gadd(circle, tv_tc);                       % line segment

%% Step 3: Observe these sequences in image space using PCA
line_segments = [pts_cs, pts_st, pts_tc];

% ApplyDimReduction(line_segments', [], 'pca')
% makeFiguresPretty;

%% Step 4: Train a multi-layer net and save as both an object and a script
architecture = [50 30 10];

if(isempty(gcp('nocreate')))
    parpool;
end
[net, tr] = Expr_LineSegments_TrainNN_fitnet(shapes, targets, architecture);
delete(gcp('nocreate'));

genFunction(net, 'NNfunction' ,'MatrixOnly','yes');
save('TrainedNet', 'net', 'tr');

%%  Pass through it some set of points on the original manifold, as well as the line segments and do PCA 
% (i.e. recreate the plot that Jascha and I were discussing)

% s= shapes(:,1);
% out1 = ExtractDataAtLayer(net, s, 2)
% out2 = NNfunction(s, 2)
% 
% out3 = ExtractDataAtLayer(net0, s, 3)
% out4 = NNfunction_nomapminmax(s, 3)

for i = 0:length(net.layers)
    layer = i;
    X = ExtractDataAtLayer(net, shapes(:, tr.testInd), layer);
    Y = vec2ind(targets(:, tr.testInd));
    X1 = [X, ExtractDataAtLayer(net, line_segments, layer)];
    Y1 = [Y, 4 * ones(1, size(line_segments,2))];

    ApplyDimReduction(X1', Y1', 'pca');
end

makeFiguresPretty;

