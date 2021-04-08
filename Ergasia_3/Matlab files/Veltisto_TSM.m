%% Nikiforidis Konstantinos 9084
% Regression
% Veltisto_TSM.m
%%
close all; 
clear all;

% fortwnw ta data
% apo tin prwti grammi kai thn 0 stili
% dioti sthn prwti grammi exw ta onomata twn features
% kanw kai kanonikopoihsh se oles tis stiles ektos apo tin target, tin 82
data = csvread('train.csv',1,0);
norm_data = data(:,1:end-1);
norm_data = normalize(norm_data);
data = [norm_data(:,1:end) data(:,end)];
% mean = mean(data);    % tsekarw an egine normalize

% Evaluation function opws kai prin gia thn R2
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% edw den diaxwrizw ta data apo twra giati tha kanw cross validation

% orizoume ton arithmo tws features kai tin aktina apo to 
% Ergasia3_2.m. Oi veltistes times me vasi tin metrikh RMSE einai oi parakato .
kept_features = 14;
aktina_r = 0.4;

% profanws pali k =5 gia to cross validation
k = 5;

% Xrisimopoiw ti sinartisi Relief gia tin meiwsei twn features
% apo matlab exw to exis
[idx,weights] = relieff(data(:,1:end-1),data(:,end),6);

% orizw pinaka gia tis metrikes
all_metrics = zeros(4);

% orizw pinaka gia na apothikeuw tis metrikes mesa ston cross
% validation. Profanos tha einai 5x4 opou 5 oi epanalipseis kai 4
% oi metrikes
metrics_of_cross_val = zeros(k,4);

% edw kanw to cross validation gia na ekpedaiusw to modelo opws kai sto
% Ergasia3_2

% cv partition gia ton k-fold
% edw xwrizw ta data se 80% train kai 20% test
part_for_kfold1 = cvpartition(data(:,end),'KFold',5,'Stratify',true);

% counter gia na metraw tis epanalipseis na mporw na kanw debug
counter = 0;

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
    
    % mf before training
    if repetition == 5
        figure();
        plotmf(my_fis,'input',1);
        titlos = "Input 1 before training";
        title(titlos);
        
        figure();
        plotmf(my_fis,'input',10);
        titlos = "Input 10 before training";
        title(titlos);
    end
    
    % edw kanw training to modelo mou me 120 epoches
    % diplasiasa to increase training rate
    [trnFis,trnError,~,valFis,valError] = anfis(training_data,my_fis,[120 0 0.01 0.9 1.1],[],checking_data);
    
    % ypologizw oles tis metrikes poy thelw
    % profanos vazw to testing data gia na testarw to modelo.
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
all_metrics(1) = sum(metrics_of_cross_val(:,1))/k;
all_metrics(2) = sum(metrics_of_cross_val(:,2))/k;
all_metrics(3) = sum(metrics_of_cross_val(:,3))/k;
all_metrics(4) = sum(metrics_of_cross_val(:,4))/k;

% PLOTS----------------------------------------------------------------

y = zeros(size(Y,1),1);
for i= 1:size(Y,1)
    y(i) = i;
end

% Times target telikoy modelou 
figure();
scatter(y,Y); grid on;
xlabel('data'); ylabel('Predicted Values');
title('Predicted Values');


% Pragmatikes times target
figure();
scatter(y,testing_data(:,end)); grid on;
xlabel('data'); ylabel('Real Values');
title('Real Values');

%Error plot in test data (prediction)
predict_error = testing_data(:,end) - Y; % oles tis grammes + tin teleutaia stili
figure();
plot(predict_error);
grid on;
xlabel('data');ylabel('Error');
titlos = " Error Veltistou Modelou ";
title(titlos);


% learning curve
% Diagramma Mathisis - Learning curve
% xrisi chk fis dioti: apo web...
% "and the tuned FIS object for which the validation error is minimum,
% chkFIS"
figure();
grid on;
plot([trnError valError]);
xlabel('# of Iterations'); ylabel('Error');
legend('Training Error','Validation Error');
titlos = "Veltisto TSK Model Learning curve";
title(titlos);

% asafi sinola sthn teliki morfi
figure();
plotmf(valFis,'input',1);
titlos = "Input 1 after training";
title(titlos);

figure();
plotmf(valFis,'input',10);
titlos = "Input 10 after training";
title(titlos);


% pinakas me times RMSE NMSE NDEI R2
% all_metrics

% Kanones
% mpainw apo to workspace kai to vlepw
number_of_rules = size(valFis.Rules,2);

