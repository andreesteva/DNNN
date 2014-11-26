function R = LowRankApproximation(M, k)
%function R = LowRankApproximation(M, k)
% calculated the rank k approximation of matrix M and returns it in matrix R
%
%   R = sum_i^k[ e_i lambda_i e_i' ] 
%   e_i are the eigenvalues with the convention that lambda_1 >= lambda_2 >= % lambda_3 ...

[V,E] = eig(M);
e = diag(E);
[e,ind] = sort(e, 'descend');
V = V(:,ind);

R = zeros(size(M));
for i = 1:k
    R = R + V(:,i)*e(i)*V(:,i)';
end