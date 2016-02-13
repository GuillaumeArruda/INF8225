clc;
clear;
load 20news_w100;
n = 4;
m = size(newsgroups,2);
o = ones(1,m);
i = 1:m;
j = newsgroups;
Y = sparse(i,j,o,m,n);

Theta = rand(4,101)-.5;
X = [documents ; ones(1,16242)];
taux_dapprentissage = 0.0005;

possibleY = eye(n);
converged = false;
yixi = Y' * X';
lastLogVraisemblance = -realmax;
[XA XV XT YA YV YT] = create_train_valid_test_splits(X,Y);
precision = [];
logVraisemblances = [];
while ~converged
    logVraisemblance = sum(sum(((Y * Theta) .* X')') - log(sum(exp(possibleY * Theta * X))));
    logVraisemblances = [logVraisemblances, logVraisemblance];
    converged = logVraisemblance - lastLogVraisemblance  < 0.1;
    lastLogVraisemblance = logVraisemblance;
    %Apprentissage
    Z = repmat(sum(exp(possibleY * Theta * XA)),4,1);
    esperance = ((exp(possibleY * Theta * XA)./Z)' * possibleY)' * XA';
    gradient = yixi - esperance;
    Theta = Theta + (taux_dapprentissage * gradient);
    
    %Validation
    Z = repmat(sum(exp(possibleY * Theta * XV)),4,1);
    reponse = exp(possibleY * Theta * XV)./Z;
    [reponse, index] = max(reponse);
    result = ones(size(reponse,2),4);
    result = result * 2;
    for i = 1 : size(index,2)
        result(i, index(i)) = 1;
    end
    d = result == YV;
    precision = [precision, sum(sum((result == YV))) / size(result,1)];
end
figure();
plot([1:size(precision,2)],precision);
figure();
plot([1:size(logVraisemblances,2)],logVraisemblances);
%Test
Z = repmat(sum(exp(possibleY * Theta * XT)),4,1);
reponse = exp(possibleY * Theta * XT)./Z;
[reponse, index] = max(reponse);
result = ones(size(reponse,2),4);
result = result * 2;
for i = 1 : size(index,2)
    result(i, index(i)) = 1;
end
precisionTest = sum(sum((result == YT))) / size(result,1);
fprintf('Precision sur lensemble de test = %f\n', full(precisionTest));

