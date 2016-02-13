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


while ~converged
    Z =  repmat(sum(exp(possibleY * Theta * X)),4,1);
    t = Y' * (Theta * X)';
    d = log(sum(exp(Theta * X)));
    logVraisemblance = sum(sum(Y' * (Theta * X)' - log(sum(sum(exp(Theta * X))))))
    esperance = ((exp(possibleY * Theta * X)./Z)' * possibleY)' * X';
    gradient = yixi - esperance;
    Theta = Theta + (taux_dapprentissage * gradient);
end
