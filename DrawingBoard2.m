% Drawing Board 2
% For when the other matlab is running Drawing Board

%%
tic;
r =rand(1200,369);
t = mat2cell(r, size(r,1), ones(1, size(r,2)));
t1 = repmat(t',1 ,length(t));
t2 = repmat(t, length(t), 1);
maxcorr = @(u,v) max(corr(u,v), corr(u,-v));
C = cellfun(maxcorr, t1, t2);
toc
%%
tic
C2 = [];
for i = 1:size(r,2)
    for j = 1:size(r,2)
        C2(i,j) = max( corr(r(:,i), r(:,j)) , corr(r(:,i), -r(:,j)) );
    end
end
toc

%% 
tic
rho1 = corr(r);
rho2 = corr(r,-r);
C3 = max(rho1,rho2);
toc