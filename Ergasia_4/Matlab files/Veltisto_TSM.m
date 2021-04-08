%% Nikiforidis Konstantinos 9084
% Regression
% Veltisto_TSM.m
%%
close all; 
clear all;

% fortwnw ta data
% apo tin prwti grammi kai thn 1 stili
% dioti sthn prwti grammi exw ta onomata twn features
% kanw kai kanonikopoihsh se oles tis stiles ektos apo tin target
data = csvread('data.csv',1,1);
norm_data = data(:,1:end-1);
norm_data = normalize(norm_data);
data = [norm_data(:,1:end) data(:,end)];
% mean = mean(data);    % tsekarw an egine normalize

% Evaluation function opws kai prin gia thn R2
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% edw den diaxwrizw ta data apo twra giati tha kanw cross validation

% orizoume ton arithmo tws features kai tin aktina apo to 
% Ergasia4_2.m. Oi veltistes times me vasi to overall accuracy .
kept_features = 11;
aktina_r = 0.2;

% Xrisimopoiw ti sinartisi Relief gia tin meiwsei twn features
% apo matlab exw to exis
[idx,weights] = relieff(data(:,1:end-1),data(:,end),6);

% pinakas gia apothikeusi olwn twn error matrix
all_error_matrix = zeros(5,5,5);
all_OA = zeros(5,1);
all_PA = zeros(5,5);    % 5 kfold kai 5 klassis
all_UA = zeros(5,5);
all_k = zeros(5,1);

% oi meses times tou telikoy modeloy
mean_error_matrix = zeros(5,5);
mean_OA = 0;
mean_PA1 = zeros(5,1);
mean_PA2 = zeros(5,1);
mean_PA3 = zeros(5,1);
mean_PA4 = zeros(5,1);
mean_PA5 = zeros(5,1);
mean_UA1 = zeros(5,1);
mean_UA2 = zeros(5,1);
mean_UA3 = zeros(5,1);
mean_UA4 = zeros(5,1);
mean_UA5 = zeros(5,1);
mean_k = 0;

% pinakas twn rule
rules_k_fold = zeros(5,1);

% cv partition gia ton k-fold
% edw xwrizw ta data se 80% train kai 20% test
part_for_kfold1 = cvpartition(data(:,end),'KFold',5,'Stratify',true);


% counter gia na metraw tis epanalipseis na mporw na kanw debug
% counter gia na doulepsei h addmf().
mf_name = strings(500000,1);
for i = 1:50000
    mf_name(i) = "mf"+i;
end
counter = 0;
counter2 = 0;

% k folds
k = 5;

% kanw k fold
for repetition = 1:part_for_kfold1.NumTestSets
    
    % disp counter
    counter2 = counter2 + 1;
    deixe ="eimai stin " + counter2 ;
    disp(deixe);
    
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
    
    % edw exw 5 klassis sinepws exw ta exis
    %%Clustering Per Class apo arxeio TSK_Classification.m
    [c1,sig1]=subclust(training_data(training_data(:,end)==1,:),aktina_r);
    [c2,sig2]=subclust(training_data(training_data(:,end)==2,:),aktina_r);
    [c3,sig3]=subclust(training_data(training_data(:,end)==3,:),aktina_r);
    [c4,sig4]=subclust(training_data(training_data(:,end)==4,:),aktina_r);
    [c5,sig5]=subclust(training_data(training_data(:,end)==5,:),aktina_r);
    num_rules=size(c1,1)+size(c2,1)+size(c3,1)+size(c4,1)+size(c5,1);
    
    %Build FIS From Scratch apo arxeio TSK_Classification.m
    my_fis=newfis('FIS_SC','sugeno');
    
    % Add Input-Output Variables
    % prepei na kanw twra mevlito names_in giati kathe fora krataw
    % kai diaforetiko arithmo features
    names_in = {};
    for i= 1:size(training_data,2)-1
        names_in{i} = "in" + i;
    end
    
    for i= 1:size(training_data,2)-1
        my_fis = addvar(my_fis,'input',names_in{i},[0 1]);
    end
    my_fis=addvar(my_fis,'output','out1',[0 1]);
    
    % Add Input Membership Functions apo arxeio TSK_Classification.m
    % 5, ena gia kathe klasi
    for i=1:size(training_data,2)-1
        for j=1:size(c1,1)
            counter = counter + 1;
            my_fis=addmf(my_fis,'input',i,mf_name(counter,1),'gaussmf',[sig1(i) c1(j,i)]);
        end
        for j=1:size(c2,1)
            counter = counter + 1;
            my_fis=addmf(my_fis,'input',i,mf_name(counter,1),'gaussmf',[sig2(i) c2(j,i)]);
        end
        for j=1:size(c3,1)
            counter = counter + 1;
            my_fis=addmf(my_fis,'input',i,mf_name(counter,1),'gaussmf',[sig3(i) c3(j,i)]);
        end
        for j=1:size(c4,1)
            counter = counter + 1;
            my_fis=addmf(my_fis,'input',i,mf_name(counter,1),'gaussmf',[sig4(i) c4(j,i)]);
        end
        for j=1:size(c5,1)
            counter = counter + 1;
            my_fis=addmf(my_fis,'input',i,mf_name(counter,1),'gaussmf',[sig5(i) c5(j,i)]);
        end
    end
    counter = 0;
    
    %Add Output Membership Functions apo arxeio TSK_Classification.m
    % edw prepei na spasw to euros [0,1] se 5 logw toy oti exw 5
    % classeis ara kanw to exis...
    params=[zeros(1,size(c1,1)) zeros(1,size(c2,1))+0.25 zeros(1,size(c3,1))+0.5 zeros(1,size(c4,1))+0.75 ones(1,size(c5,1))];
    for i=1:num_rules
        counter = counter + 1;
        my_fis=addmf(my_fis,'output',1,mf_name(counter,1),'constant',params(i));
    end
    counter = 0;
    
    %Add FIS Rule Base apo arxeio TSK_Classification.m
    ruleList=zeros(num_rules,size(training_data,2));
    for i=1:size(ruleList,1)
        ruleList(i,:)=i;
    end
    ruleList=[ruleList ones(num_rules,2)];
    my_fis=addrule(my_fis,ruleList);
    
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
   
    %Train & Evaluate ANFIS apo arxeio TSK_Classification.m
    % training gia 100 epoches
    [trnFis,trnError,~,valFis,valError]=anfis(training_data,my_fis,[100 0 0.01 0.9 1.1],[],checking_data);
    %             figure();
    %             plot([trnError valError],'LineWidth',2); grid on;
    %             legend('Training Error','Validation Error');
    %             xlabel('# of Epochs');
    %             ylabel('Error');
    %             titlos = "Class Dependent me aktina_r = " + aktina_r(k);
    %             title(titlos);
    Y=evalfis(testing_data(:,1:end-1),valFis);
    Y=round(Y);
    % thelw mono times 1,2,3,4,5 opws oi klaseis mou
    for i=1:size(Y,1)
        if Y(i) < 1
            Y(i) = 1;
        elseif Y(i) > 5
            Y(i) = 5;
        end
    end
    diff=testing_data(:,end)-Y;
    
    
    
    Error_matrix = zeros(5); % afoy exw 5 klaseis
    Error_matrix = confusionmat(testing_data(:,end),Y);
    N = size(testing_data,1);
    OA = sum(diag(Error_matrix))/N;
    rules_k_fold(repetition,1) = size(valFis.Rules,2);
    
    all_error_matrix(:,:,repetition) = Error_matrix;
    all_OA(repetition) = OA;
    all_PA(repetition,1) = all_error_matrix(1,1,repetition)/sum(all_error_matrix(1,:,repetition));
    all_PA(repetition,2) = all_error_matrix(2,2,repetition)/sum(all_error_matrix(2,:,repetition));
    all_PA(repetition,3) = all_error_matrix(3,3,repetition)/sum(all_error_matrix(3,:,repetition));
    all_PA(repetition,4) = all_error_matrix(4,4,repetition)/sum(all_error_matrix(4,:,repetition));
    all_PA(repetition,5) = all_error_matrix(5,5,repetition)/sum(all_error_matrix(5,:,repetition));
    all_UA(repetition,1) = all_error_matrix(1,1,repetition)/sum(all_error_matrix(:,1,repetition));
    all_UA(repetition,2) = all_error_matrix(2,2,repetition)/sum(all_error_matrix(:,2,repetition));
    all_UA(repetition,3) = all_error_matrix(3,3,repetition)/sum(all_error_matrix(:,3,repetition));
    all_UA(repetition,4) = all_error_matrix(4,4,repetition)/sum(all_error_matrix(:,4,repetition));
    all_UA(repetition,5) = all_error_matrix(5,5,repetition)/sum(all_error_matrix(:,5,repetition));
    c = 0;
    for i = 1:5
        c = c + sum(all_error_matrix(:,i,repetition))*sum(all_error_matrix(i,:,repetition));
    end
    all_k(repetition) = (N*sum(diag(all_error_matrix(:,:,repetition))) - c)/(N^2 - c);
    
end

for i = 1:k
    mean_error_matrix = mean_error_matrix + all_error_matrix(:,:,i);
end
mean_error_matrix = mean_error_matrix/k;
mean_OA = sum(all_OA)/k;
mean_PA1 = sum(all_PA(:,1))/k;
mean_PA2 = sum(all_PA(:,2))/k;
mean_PA3 = sum(all_PA(:,3))/k;
mean_PA4 = sum(all_PA(:,4))/k;
mean_PA5 = sum(all_PA(:,5))/k;
mean_UA1 = sum(all_UA(:,1))/k;
mean_UA2 = sum(all_UA(:,2))/k;
mean_UA3 = sum(all_UA(:,3))/k;
mean_UA4 = sum(all_UA(:,4))/k;
mean_UA5 = sum(all_UA(:,5))/k;
mean_k = sum(all_k)/k;
mean_rules = sum(rules_k_fold)/k;

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
