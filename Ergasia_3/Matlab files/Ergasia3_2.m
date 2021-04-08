%% Nikiforidis Konstantinos 9084
% Regression
% Ergasia3_2.m
%%
close all; 
clear all;

% fortwnw ta data
% apo tin prwti grammi kai thn 0 stili
% dioti sthn prwti grammi exw ta onomata twn features
% kanw kai kanonikopoihsh se oles tis stiles ektos apo tin target, diladi tin 82
data = csvread('train.csv',1,0);
norm_data = data(:,1:end-1);
norm_data = normalize(norm_data);
data = [norm_data(:,1:end) data(:,end)];
% mean = mean(data);    % tsekarw an egine normalize

% Evaluation function opws kai prin gia thn R2
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% edw den diaxwrizw ta data apo twra giati tha kanw cross validation

% grid searching method
% tha dokimasoyme tis eksis times gia ta features poy tha krathsoye
% 5 8 11 14
% kai tis eksis times tis aktinas r
% 0.2 0.4 0.6 0.8 1
parameters = zeros(4,5,2); %trisdiastatos gia na orisw tis times pou thelw
% einai 4x5x2 giati exw 4x5 dokimes kai x2 giati exw 2 parametrous pou tha
% elegxw
% ara

parameters(:,:,1) = [5 5 5 5 5; 8 8 8 8 8; 11 11 11 11 11; 14 14 14 14 14];
parameters(:,:,2) = [0.2 0.4 0.6 0.8 1; 0.2 0.4 0.6 0.8 1; 0.2 0.4 0.6 0.8 1; 0.2 0.4 0.6 0.8 1];

% ftiaxnw allon enan pinaka me diastasis 4x5x4 wste na apothikeusw kai oles
% tis metrikes gia tis epanalipseis pou tha kanw 
all_metrics = zeros(4,5,4);

% orizw to k gia ton k-fold cross validation pou tha ginei parakatw
k = 5;

% Xrisimopoiw ti sinartisi Relief gia tin meiwsei twn features
% apo matlab exw to exis
[idx,weights] = relieff(data(:,1:end-1),data(:,end),6);

% pinakas rmse monodiastatos gia plot sfalma arithmos kanonwn
% kai sfalma- epilexthentwn xarakt
rmse = zeros(4,5);
rules = zeros(4,5);
kept_f = zeros(4,5);


% counter gia na metraw tis epanalipseis na mporw na kanw debug
counter = 0;
% pame twra na kanoume to grid search
for i = 1:4
    for j = 1:5
        kept_features = parameters(i,j,1);
        aktina_r = parameters(i,j,2);
        % cv partition gia ton k-fold
        % edw xwrizw ta data se 80% train kai 20% test
        part_for_kfold1 = cvpartition(data(:,end),'KFold',5,'Stratify',true);
        
        % orizw pinaka gia na apothikeuw tis metrikes mesa ston cross
        % validation. Profanos tha einai 5x4 opou 5 oi epanalipseis kai 4
        % oi metrikes
        metrics_of_cross_val = zeros(k,4);
        
        
        for repetition = 1:part_for_kfold1.NumTestSets
           % edw xwrizw to training data apo to cross validation
           % pou htan to 80% tou arxikoy data, se 60% training kai 20%
           % check data
           endiamesa_training_data = data(training(part_for_kfold1,repetition),:);
           testing_data = data(test(part_for_kfold1,repetition),:);
           % twra xwrizw endiamesa_training_data se 60% training_data kai
           % 20% testing data
           part_for_kfold2 = cvpartition(endiamesa_training_data(:,end),'KFold',4,'Stratify',true);
           training_data = endiamesa_training_data(training(part_for_kfold2,2),:);
           checking_data = endiamesa_training_data(test(part_for_kfold2,2),:);
           
           % edw prepei na kratisw ta features pou analogoun sti kathe
           % epanalipsi sto kathe set dedomenwn pou exw, sinepws exw ta exis
           training_data = [training_data(:, idx(1:kept_features)) training_data(:,end)];
           checking_data = [checking_data(:, idx(1:kept_features)) checking_data(:,end)];
           testing_data = [testing_data(:, idx(1:kept_features)) testing_data(:,end)];
           
           % orizw to fis
           my_fis = genfis2(training_data(:,1:end-1), training_data(:,end), aktina_r);
           
           % edw kanw training to modelo mou me 100 epoches
           [trnFis,trnError,~,valFis,valError] = anfis(training_data,my_fis,[100 0 0.01 0.9 1.1],[],checking_data);
           
           % ypologizw oles tis metrikes poy thelw
           % profanos vazw to test gia na testarw to modelo.
           Y = evalfis(testing_data(:,1:end-1),valFis); 
           R2 = Rsq(Y,testing_data(:,end));
           RMSE = sqrt(mse(Y,testing_data(:,end)));
           NMSE = 1 - R2; % R2 = 1 - NMSE
           NDEI = sqrt(NMSE);
           
           % ola ta error apo to crossvalidation pou meta tha vrw ton meso
           % oro tous gia na ta sigkrinw sto telos.
           metrics_of_cross_val(repetition,1) = R2;
           metrics_of_cross_val(repetition,2) = RMSE;
           metrics_of_cross_val(repetition,3) = NMSE;
           metrics_of_cross_val(repetition,4) = NDEI;
           
           % disp counter
           counter = counter + 1;
           deixe ="eimai stin " + counter ;
           disp(deixe);
           
        end
        
        % Ipologizw oles tis meses metrikes meta to cross validation
        % k = 5 profanws afou exw 5 fold cross validation
        all_metrics(i,j,1) = sum(metrics_of_cross_val(:,1))/k;
        all_metrics(i,j,2) = sum(metrics_of_cross_val(:,2))/k;
        all_metrics(i,j,3) = sum(metrics_of_cross_val(:,3))/k;
        all_metrics(i,j,4) = sum(metrics_of_cross_val(:,4))/k;
        
        % apothikeusi rmse gia plots
        rmse(i,j) = all_metrics(i,j,2);
        rules(i,j) = size(valFis.Rules,2);
        kept_f(i,j) = kept_features;

    end
end

% PLOTS
% sfalma se sxesi me kanones
figure();
scatter(reshape(rmse,1,[]),reshape(rules,1,[])); grid on;
xlabel("RMSE"); 
ylabel("Number of Rules");
title("RMSE relevant to Number of Rules ");

% sfalma se sxesi me kratimena features
figure();
scatter(reshape(rmse,1,[]),reshape(kept_f,1,[])); grid on;
xlabel("RMSE"); 
ylabel("Number of kept features");
title("RMSE relevant to Number of kept features ");

% sfalma se sxesi me tin aktina_r
figure();
scatter(reshape(rmse,1,[]),reshape(parameters(:,:,2),1,[])); grid on;
xlabel("RMSE"); 
ylabel("Aktina cluster");
title("RMSE relevant to Aktina cluster ");

% epifaneia sfalmatos sxetika me aktina kai kept features
figure();
surf(all_metrics(:,:,2),parameters(:,:,2),parameters(:,:,1)); grid on;
xlabel("Sfalma"); ylabel("aktina_r"); zlabel("Number of features");
title("Surface of RMSE relevant to aktina_r and Number of features.");

% ara to modelo pou kratame einai me aktina 0.4 kai kept features
% 14.