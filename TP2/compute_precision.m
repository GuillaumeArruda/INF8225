function [ precision ] = compute_precision( X, Y, Theta)
    possibleY = eye(size(Y,2));
    Z = repmat(sum(exp(possibleY * Theta * X)),4,1);
    reponse = exp(possibleY * Theta * X)./Z;
    [reponse, index] = max(reponse);
    result = ones(size(reponse,2),4);
    result = result * 2;
    for i = 1 : size(index,2)
        result(i, index(i)) = 1;
    end
    precision = sum(sum((result == Y))) / size(result,1);
end

