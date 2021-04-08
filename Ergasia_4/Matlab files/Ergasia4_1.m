%% Nikiforidis Konstantinos 9084
% Classification
% Ergasia4_1.m
%%
close all; 
clear all;

% fortwnw ta data
data = load('haberman.data');
preproc = 1;

% xwrizw ta data me splitscale se 60 20 20 
[trnData,chkData,tstData]=split_scale(data,preproc);   % opws sto arxeio TSK_Regression

% vlepontas ta data , exoume 2 klaseis, thn 1 kai thn 2.

% episis h metavliti pou tha kathorizei ton arithmo twn asafwn kanonwn
% profanws kai einai h aktina epirohs twn cluster. Sinepws opws ma zhtaei h
% ekfwnisi tha xrisimopoihsoume 2 akraies times to 0.2 kai to 0.9 wste na
% exoume pio katanoita apotelesmata.
aktina_r = [0.2 0.9];

% pinakes gia na apothikeusw tis metriseis gia dependend
all_OA_dep = zeros(2,1);
all_PA_dep = zeros(2,2);
all_UA_dep = zeros(2,2);
all_K_dep = zeros(2,1);
all_Error_matrix_dep = zeros(2,2,2);
all_rules_dep = zeros(2,1);

% pinakes gia na apothikeusw tis metriseis gia independed
all_OA_indep = zeros(2,1);
all_PA_indep = zeros(2,2);
all_UA_indep = zeros(2,2);
all_K_indep = zeros(2,1);
all_Error_matrix_indep = zeros(2,2,2);
all_rules_indep = zeros(2,1);

% counter gia na doulepsei h addmf().
mf_name = strings(10000,1);
for i = 1:10000
    mf_name(i) = "mf"+i;
end
counter = 0;

% tha trexw twra gia thn kathe aktina 2 montela kathe fora
% ena gia class independent kai ena gia class dependent
for k = 1:2
    
    %%Clustering Per Class apo arxeio TSK_Classification.m
    [c1,sig1]=subclust(trnData(trnData(:,end)==1,:),aktina_r(k));
    [c2,sig2]=subclust(trnData(trnData(:,end)==2,:),aktina_r(k));
    num_rules=size(c1,1)+size(c2,1);
    
    %Build FIS From Scratch apo arxeio TSK_Classification.m
    my_fis=newfis('FIS_SC','sugeno');
    
    %Add Input-Output Variables apo arxeio TSK_Classification.m
    names_in={'in1','in2','in3'};
    for i=1:size(trnData,2)-1
        my_fis=addvar(my_fis,'input',names_in{i},[0 1]);
    end
    my_fis=addvar(my_fis,'output','out1',[0 1]);
    
    %Add Input Membership Functions apo arxeio TSK_Classification.m
    name = 'sth';
    for i=1:size(trnData,2)-1
        for j=1:size(c1,1)
            counter = counter + 1;
            my_fis=addmf(my_fis,'input',i,mf_name(counter,1),'gaussmf',[sig1(i) c1(j,i)]);
        end
        for j=1:size(c2,1)
            counter = counter + 1;
            my_fis=addmf(my_fis,'input',i,mf_name(counter,1),'gaussmf',[sig2(i) c2(j,i)]);
        end
    end
    
    counter = 0;
    %Add Output Membership Functions apo arxeio TSK_Classification.m
    params=[zeros(1,size(c1,1)) ones(1,size(c2,1))];
    for i=1:num_rules
        counter = counter + 1;
        my_fis=addmf(my_fis,'output',1,mf_name(counter,1),'constant',params(i));
    end
    counter = 0;
    %Add FIS Rule Base apo arxeio TSK_Classification.m
    ruleList=zeros(num_rules,size(trnData,2));
    for i=1:size(ruleList,1)
        ruleList(i,:)=i;
    end
    ruleList=[ruleList ones(num_rules,2)];
    my_fis=addrule(my_fis,ruleList);
    
    %Train & Evaluate ANFIS apo arxeio TSK_Classification.m
    % training gia 100 epoches
    [trnFis,trnError,~,valFis,valError]=anfis(trnData,my_fis,[100 0 0.01 0.9 1.1],[],chkData);
    figure();
    plot([trnError valError],'LineWidth',2); grid on;
    legend('Training Error','Validation Error');
    xlabel('# of Epochs');
    ylabel('Error');
    titlos = "Class Dependent me aktina_r = " + aktina_r(k);
    title(titlos);
    Y=evalfis(tstData(:,1:end-1),valFis);
    Y=round(Y);
    % thelw mono times 1 kai 2 opws oi klaseis mou
    for i=1:size(Y,1)
        if Y(i) < 1
            Y(i) = 1;
        elseif Y(i) > 2
            Y(i) = 2;
        end
    end
    
    diff=tstData(:,end)-Y;
    % Acc=(length(diff)-nnz(diff))/length(Y)*100;
    
    % plotarw tis sinartisis simmetoxis pou xreiazomai gia kathe feature
    for j = 1:size(trnData,2)-1 % osa einai kai ta features
        figure();
        plotmf(valFis,'input',j);
        titlos = "Model dependent " + k + " Feature " + j + " me aktina_r = " + aktina_r(k);
        title(titlos);
    end
    
    % gia na mporesw na ipologisw oles tis metrikes prepei na ftiaxw ton Error
    % matrix opws leei i ekfwnisi
    Error_matrix = zeros(2); % afoy exw 2 klaseis
    Error_matrix = confusionmat(tstData(:,end),Y);
    
    % overall accuracy
    TP = Error_matrix(1,1);
    FP = Error_matrix(2,1);
    TN = Error_matrix(2,2);
    FN = Error_matrix(1,2);
    OA = (TP + TN)/(TP + FP + TN + FN);
    
    % PA UA
    PA1 = TP/(FN + TP);
    PA2 = TN/(FP + TN);
    UA1 = TP/(TP + FP);
    UA2 = TN/(TN + FN);
    
    % K
    N = size(tstData,1);
    K = (N*(TP + TN) - ((TP + FP)*(TP + FN) + (FN + TN)*(FP + TN)))/(N^2 - ((TP + FP)*(TP + FN) + (FN + TN)*(FP + TN)));
    
    % apothikeuw tis metriseis tis dep
    all_OA_dep(k,1) = OA;
    all_PA_dep(k,1) = PA1;
    all_PA_dep(k,2) = PA2;
    all_UA_dep(k,1) = UA1;
    all_UA_dep(k,2) = UA2;
    all_K_dep(k,1) = K;
    all_Error_matrix_dep(:,:,k) = Error_matrix;
    
    % arithmos kanonwn
    all_rules_dep(k,1) = size(valFis.Rules,2);
    
    
    %Compare with Class-Independent Scatter Partition apo arxeio TSK_Classification.m
    my_fis2=genfis2(trnData(:,1:end-1),trnData(:,end),aktina_r(k));
    [trnFis,trnError,~,valFis,valError]=anfis(trnData,my_fis2,[100 0 0.01 0.9 1.1],[],chkData);
    
    figure();
    plot([trnError valError],'LineWidth',2); grid on;
    legend('Training Error','Validation Error');
    xlabel('# of Epochs');
    ylabel('Error');
    titlos = "Class Independent me aktina_r = " + aktina_r(k) ;
    title(titlos);
    
    Y=evalfis(tstData(:,1:end-1),valFis);
    Y=round(Y);
    % thelw mono times 1 kai 2 opws oi klaseis mou
    for i=1:size(Y,1)
        if Y(i) < 1
            Y(i) = 1;
        elseif Y(i) > 2
            Y(i) = 2;
        end
    end
    
    diff=tstData(:,end)-Y;
    
    % plotarw tis sinartisis simmetoxis pou xreiazomai gia kathe feature
    for j = 1:size(trnData,2)-1 % osa einai kai ta features
        figure();
        plotmf(valFis,'input',j);
        titlos = "Model independent " + k + " Feature " + j + " me aktina_r = " + aktina_r(k) ;
        title(titlos);
    end
    
    % metrikes gia independent
    Error_matrix = confusionmat(tstData(:,end),Y);
    
    % overall accuracy
    TP = Error_matrix(1,1);
    FP = Error_matrix(2,1);
    TN = Error_matrix(2,2);
    FN = Error_matrix(1,2);
    OA = (TP + TN)/(TP + FP + TN + FN);
    
    % PA UA
    PA1 = TP/(FN + TP);
    PA2 = TN/(FP + TN);
    UA1 = TP/(TP + FP);
    UA2 = TN/(TN + FN);
    
    % K
    N = size(tstData,1);
    K = (N*(TP + TN) - ((TP + FP)*(TP + FN) + (FN + TN)*(FP + TN)))/(N^2 - ((TP + FP)*(TP + FN) + (FN + TN)*(FP + TN)));
    
    % apothikeuw tis metriseis tis dep
    all_OA_indep(k,1) = OA;
    all_PA_indep(k,1) = PA1;
    all_PA_indep(k,2) = PA2;
    all_UA_indep(k,1) = UA1;
    all_UA_indep(k,2) = UA2;
    all_K_indep(k,1) = K;
    all_Error_matrix_indep(:,:,k) = Error_matrix;
    
    % arithmos kanonwn
    all_rules_indep(k,1) = size(valFis.Rules,2);
    
end

% emfanisi pinakwn
all_OA_dep 
all_PA_dep 
all_UA_dep 
all_K_dep 
all_Error_matrix_dep 
all_rules_dep

all_OA_indep 
all_PA_indep 
all_UA_indep 
all_K_indep 
all_Error_matrix_indep 
all_rules_indep






