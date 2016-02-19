clc;
clear;
load 20news_w100;
n = 4;
m = size(newsgroups,2);
o = ones(1,m);
i = 1:m;
j = newsgroups;
Y = sparse(i,j,o,m,n);


Theta = rand(4,201)-.5;
X = [documents ;randi(2,100,16242) - 1; ones(1,16242)];
[XA XV XT YA YV YT] = create_train_valid_test_splits(X,Y);
%Approche sans regulation
deltaTheta = zeros(4,201);
alpha = 0.5;
batchSize = 568;
nbIteration = 1;
mblogVraisemblances = [];
lastLogVraisemblance = -realmax;
bmaprecision = [];
bmvprecision = [];
converged = false;
taux_dapprentissage = 0.0001;
temps = 1;
possibleY = eye(4);
while ~converged
    [XBatch YBatch] = create_mini_batches(XA,YA,batchSize);
    taux_dapprentissage = 2/temps;
    for i = 1:size(XBatch,2)
        logVraisemblance = sum(sum(((YV * Theta) .* XV')') - log(sum(exp(possibleY * Theta * XV))));
        mblogVraisemblances = [mblogVraisemblances, logVraisemblance];
        converged = abs(logVraisemblance - lastLogVraisemblance)  < 0.001;
        if(converged)
            break;
        end
        lastLogVraisemblance = logVraisemblance;
        %Apprentissage
        Z = repmat(sum(exp(possibleY * Theta * XBatch{:,i})),4,1);
        esperance = ((exp(possibleY * Theta * XBatch{:,i})./Z) * XBatch{:,i}');
        yixi = YBatch{i,:}' * XBatch{:,i}';
        gradient = (yixi - esperance) ./batchSize;
        deltaTheta = alpha*deltaTheta + taux_dapprentissage * gradient;
        Theta = Theta + deltaTheta;
        bmvprecision = [bmvprecision, compute_precision(XV,YV,Theta)];
    end
    bmaprecision = [bmaprecision, compute_precision(XA,YA,Theta)];
    temps = temps + 1;
end

figure();
subplot(2,2,1);
histogram(abs(Theta(1,1:101)), 20);
title({'Histogramme des paramètres valides (Y = 1)';'sans terme de régularisation'});
xlabel('Valeur absolue du poid');
ylabel('Occurences');
subplot(2,2,2);
histogram(abs(Theta(2,1:101)), 20);
title({'Histogramme des paramètres valides (Y = 2)';'sans terme de régularisation'});
xlabel('Valeur absolue du poid');
ylabel('Occurences');
subplot(2,2,3);
histogram(abs(Theta(3,1:101)), 20);
title({'Histogramme des paramètres valides (Y = 3)';'sans terme de régularisation'});
xlabel('Valeur absolue du poid');
ylabel('Occurences');
subplot(2,2,4);
histogram(abs(Theta(4,1:101)), 20);
title({'Histogramme des paramètres valides (Y = 4)';'sans terme de régularisation'});
xlabel('Valeur absolue du poid');
ylabel('Occurences');

figure();
histogram(abs(Theta(:,102:end)), 20);
title({'Histogramme des paramètres valides';'sans terme de régularisation'});
xlabel('Valeur absolue du poid');
ylabel('Occurences');


lastLogVraisemblance = -realmax;
deltaTheta = zeros(4,201);
alpha = 0.5;
batchSize = 568;
nbIteration = 1;
mblogVraisemblances = [];
lastLogVraisemblance = -realmax;
converged = false;
taux_dapprentissage = 0.1;
temps = 1;
lambda2 = 0.016;
lambda1 = 0.046;
while ~converged
    [XBatch YBatch] = create_mini_batches(XA,YA,batchSize);
    taux_dapprentissage = 2/temps;
    for i = 1:size(XBatch,2)
        logVraisemblance = sum(sum(((YV * Theta) .* XV')') - log(sum(exp(possibleY * Theta * XV))));
        mblogVraisemblances = [mblogVraisemblances, logVraisemblance];
        converged = abs(logVraisemblance - lastLogVraisemblance)  < 0.001 || isnan(logVraisemblance);
        if(converged )
            break;
        end
        lastLogVraisemblance = logVraisemblance;
        %Apprentissage
        Z = repmat(sum(exp(possibleY * Theta * XBatch{:,i})),4,1);
        esperance = ((exp(possibleY * Theta * XBatch{:,i})./Z) * XBatch{:,i}');
        yixi = YBatch{i,:}' * XBatch{:,i}';
        gradient = ((yixi - esperance) ./batchSize) - ((batchSize / size(XA,2)) * ((lambda2 * 2 * Theta) + (lambda1 * ((Theta > 0) + (Theta < 0) * -1))));
        deltaTheta = alpha*deltaTheta + taux_dapprentissage * gradient;
        Theta = Theta + deltaTheta;
        compute_precision(XV,YV,Theta);
        bmvprecision = [bmvprecision, compute_precision(XV,YV,Theta)];
    end
    bmaprecision = [bmaprecision, compute_precision(XA,YA,Theta)];
    temps = temps + 1;
end

figure();
subplot(2,2,1);
histogram(abs(Theta(1,1:101)), 20);
title({'Histogramme des paramètres valides (Y = 1)';'avec terme de régularisation'});
xlabel('Valeur absolue du poid');
ylabel('Occurences');
subplot(2,2,2);
histogram(abs(Theta(2,1:101)), 20);
title({'Histogramme des paramètres valides (Y = 2)';'avec terme de régularisation'});
xlabel('Valeur absolue du poid');
ylabel('Occurences');
subplot(2,2,3);
histogram(abs(Theta(3,1:101)), 20);
title({'Histogramme des paramètres valides (Y = 3)';'avec terme de régularisation'});
xlabel('Valeur absolue du poid');
ylabel('Occurences');
subplot(2,2,4);
histogram(abs(Theta(4,1:101)), 20);
title({'Histogramme des paramètres valides (Y = 4)';'avec terme de régularisation'});
xlabel('Valeur absolue du poid');
ylabel('Occurences');
figure();
histogram(abs(Theta(:,102:end)), 20);
title({'Histogramme des paramètres aléatoires';'avec terme de régularisation'});
xlabel('Valeur absolue du poid');
ylabel('Occurences');

precisionTest = compute_precision(XT,YT, Theta);
fprintf('Precision sur lensemble de test = %f\n', full(precisionTest));