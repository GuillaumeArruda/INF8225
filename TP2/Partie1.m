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

converged = false;
yixi = Y' * X';
while ~converged
    WX = Theta * X;
    logSumExpWX = log(sum(exp(WX)));
    logVraisemblance = sum(sum(Y * bsxfun(@minus,WX,logSumExpWX))); 
end