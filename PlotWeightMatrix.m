function PlotWeightMatrix(w)
    % format: outputs x inputs
    % i.e. a 3 x 900 matrix will create a figure with 3 subplots each
    % containing a 30 x 30 image

    % Find good subplot dimensions to use            
    [s1, s2] = findIntegerFactorsCloseToSquareRoot(size(w,1));

    % Find Image Dimensions
    [a, b] = findIntegerFactorsCloseToSquareRoot(size(w,2));

    % Plot
    figure;
    for i =1:s1*s2
        subplot(s1, s2, i);
        imagesc(reshape(w(i,:), a,b));
        colormap('gray');
    %     colorbar;
    end

end

function [a, b] =  findIntegerFactorsCloseToSquareRoot(n)
    % a cannot be greater than the square root of n
    % b cannot be smaller than the square root of n
    % we get the maximum allowed value of a
    amax = floor(sqrt(n));
    if 0 == rem(n, amax)
        % special case where n is a square number
        a = amax;
        b = n / a;
        return;
    end
    % Get its prime factors of n
    primeFactors  = factor(n);
    % Start with a factor 1 in the list of candidates for a
    candidates = [1];
    for i=1:numel(primeFactors)
        % get the next prime factr
        f = primeFactors(i);
        % Add new candidates which are obtained by multiplying
        % existing candidates with the new prime factor f
        % Set union ensures that duplicate candidates are removed
        candidates  = union(candidates, f .* candidates);
        % throw out candidates which are larger than amax
        candidates(candidates > amax) = [];
    end
    % Take the largest factor in the list d
    a = candidates(end);
    b = n / a;
end