%% Nikiforidis Konstantinos 9084
% Regression
% Ergasia3_1.m
%%
close all; 
clear all;

% fortwnw ta data
data = load('airfoil_self_noise.dat');
preproc=1;
% scale ta data
[trnData,chkData,tstData]=split_scale(data,preproc);   % opws sto arxeio TSK_Regression
Perf=zeros(4,4); % arxikopoihsh toy pinaka me tis metrikes

% Evaluation function
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% FIS with grid partition, opws fainontai ston pinaka 1 sthn ekfwnisi
% edw ftiaxnetai pinakas apo object sugfis me 4 stoixeia
% osa diladi einai kai ta montela pou prepei na ilopoihsoume
% gia na mporesoume na kanoume kai ta 4 montelo me mia for
% kai oxi me 4 diaforetika arxeia
myfis(1)=genfis1(trnData,2,'gbellmf','constant');   % 1 
myfis(2)=genfis1(trnData,3,'gbellmf','constant');   % 2
myfis(3)=genfis1(trnData,2,'gbellmf','linear');     % 3 
myfis(4)=genfis1(trnData,3,'gbellmf','linear');     %4

counter = 0;


% ekpaideuw ta modela mou kai plotarw kiolas
for i = 1:4
    % anfis
    % to trexw gia 100 epoches to kathena
    [trnFis,trnError,~,valFis,valError] = anfis(trnData,myfis(i),[100 0 0.01 0.9 1.1],[],chkData);
    
    % disp counter
    counter = counter + 1;
    deixe ="eimai stin " + counter ;
    disp(deixe);
    
    % Membership functions plots
    for j = 1:5 % osa einai kai ta features
        figure();
        plotmf(valFis,'input',j);
        titlos = "Model " + i + " Feature " + j;
        title(titlos);
    end
    
    % Diagramma Mathisis - Learning curve
    % xrisi chk fis dioti
    % and the tuned FIS object for which the validation error is minimum, chkFIS
    figure();
    grid on;
    plot([trnError valError]);
    xlabel('# of Iterations'); ylabel('Error');
    legend('Training Error','Validation Error');
    titlos = "Model " + i + " Learning Curve ";
    title(titlos);
    
    % ypologizw oles tis metrikes poy thelw
    % profanos vazw to tstData gia na testarw to modelo.
    Y = evalfis(tstData(:,1:end-1),valFis); % 1 eos 5 osa ta features
    R2 = Rsq(Y,tstData(:,end));
    RMSE = sqrt(mse(Y,tstData(:,end)));
    NMSE = 1 - R2; % R2 = 1 - NMSE
    NDEI = sqrt(NMSE);
    Perf(:,i) = [R2; RMSE; NMSE; NDEI];
    
    %Error plot in test data (prediction)
    predict_error = tstData(:,end) - Y; % oles tis grammes + tin teleutaia stili
    figure();
    plot(predict_error);
    grid on;
    xlabel('input');ylabel('Error');
    titlos = "Model " + i + " Prediction Error ";
    title(titlos);
end

% Results Table
varnames={'Model1', 'Model2', 'Model3', 'Model4'};
rownames={'Rsquared' , 'RMSE' , 'NMSE' , 'NDEI'};
Perf = array2table(Perf,'VariableNames',varnames,'RowNames',rownames);


