%% Nikiforidis Konstantinos 9084
% Classification
% Ergasia4_2.m
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

% grid searching method
% tha dokimasoyme tis eksis times gia ta features poy tha krathsoyme
% 5 7 9 11, kathos gia parapanw features to pc ftanei tavani kai kanei
% meres mono gia 1 fold cross validation.
% kai tis eksis times tis aktinas r
% 0.2 0.4 0.6 0.8 1
parameters = zeros(4,5,2); %trisdiastatos gia na orisw tis times pou thelw
% einai 4x5x2 giati exw 4x5 dokimes kai x2 giati exw 2 parametrous pou tha
% elegxw
% ara

parameters(:,:,1) = [5 5 5 5 5; 7 7 7 7 7; 9 9 9 9 9; 11 11 11 11 11];
parameters(:,:,2) = [0.2 0.4 0.6 0.8 1; 0.2 0.4 0.6 0.8 1; 0.2 0.4 0.6 0.8 1; 0.2 0.4 0.6 0.8 1];


% orizw to k gia ton k-fold cross validation pou tha ginei parakatw
k = 5;

% Xrisimopoiw ti sinartisi Relief gia tin meiwsei twn features
% apo matlab exw to exis
[idx,weights] = relieff(data(:,1:end-1),data(:,end),6);

% counter gia na metraw tis epanalipseis na mporw na kanw debug
% counter gia na doulepsei h addmf().
mf_name = strings(500000,1);
for i = 1:50000
    mf_name(i) = "mf"+i;
end
counter = 0;
counter2 = 0;

% pinakas gia na apothikeuw ston k-fold to OA
OA_k_fold = zeros(5,1);
rules_k_fold = zeros(5,1);

% telikos pinakas me ola ta OA
all_OA = zeros(4,5);
rules = zeros(4,5);
kept_f = zeros(4,5);

% pame twra na kanoume to grid search me class dependent
for p = 1:4
    for q = 1:5
        
        
        
        kept_features = parameters(p,q,1);
        aktina_r = parameters(p,q,2);
        % cv partition gia ton k-fold
        % edw xwrizw ta data se 80% train kai 20% test
        part_for_kfold1 = cvpartition(data(:,end),'KFold',5,'Stratify',true);
        
        % orizw pinaka gia na apothikeuw tis metrikes mesa ston cross
        % validation. Profanos tha einai 5x4 opou 5 oi epanalipseis kai 4
        % oi metrikes
        metrics_of_cross_val = zeros(k,4);
        
        
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
            OA_k_fold(repetition,1) = OA;
            rules_k_fold(repetition,1) = size(valFis.Rules,2);
            
   
            
        end
        
        % genika to meso sfalma de me fainetai kali epilogi gia metriki se
        % provlhma classification, tha xrshimopoihsw ws metrikh gia ton
        % diaxwrismo mou, thn metrikh toy overall accuracy.
        % Vriskw to meso OA meta ton kathe k-fold
        all_OA(p,q) = sum(OA_k_fold(:,1))/k;
        kept_f(p,q) = kept_features;
        rules(p,q) = sum(rules_k_fold(:,1))/k;
        
    end
end


% PLOTS
% OA se sxesi me kanones
figure();
scatter(reshape(all_OA,1,[]),reshape(rules,1,[])); grid on;
xlabel("Overall Accuracy"); 
ylabel("Number of Rules");
title("Overall Accuracy relevant to Number of Rules ");

% OA se sxesi me kratimena features
figure();
scatter(reshape(all_OA,1,[]),reshape(kept_f,1,[])); grid on;
xlabel("Overall Accuracy"); 
ylabel("Number of kept features");
title("Overall Accuracy relevant to Number of kept features ");

% OA se sxesi me tin aktina_r
figure();
scatter(reshape(all_OA,1,[]),reshape(parameters(:,:,2),1,[])); grid on;
xlabel("Overall Accuracy"); 
ylabel("Aktina cluster");
title("Overall Accuracy relevant to Aktina cluster ");

% epifaneia OA sxetika me aktina kai kept features
figure();
surf(all_OA(:,:),parameters(:,:,2),parameters(:,:,1)); grid on;
xlabel("Overall Accuracy"); ylabel("aktina_r"); zlabel("Number of features");
title("Surface of OA relevant to aktina_r and Number of features.");



