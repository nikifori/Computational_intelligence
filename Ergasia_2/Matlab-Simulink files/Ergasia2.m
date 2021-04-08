%% Nikiforidis Konstantinos 9084
% F_CarControl
% Ergasia2.m
%%
close all; 
clear all;

Car_FLC = readfis('Car_FLC_teliko_withrules');
fuzzyLogicDesigner(Car_FLC);

% prwta trexo to simulink
% exw vali block poy pairnei ta X kai Y apo to kiklwma kai ta pernaei sto
% workspace.
% meta ta plotarw gia kathe gonia ekkinisis
% plot

% figure();
% plot(out.X,out.Y);
% xlabel("x"); ylabel("y");
% grid on;
% legend('kinisi autokinitou');
% title("Kinisi Autokinitou gia 0 moires arxiki timi");
% xlim([0 10]);

% edw plotarw ta empodia.
% points

% x1 = [5 5];
% y1 = [0 1];
% x2 = [5 6];
% y2 = [1 1];
% x3 = [6 6];
% y3 = [1 2];
% x4 = [6 7];
% y4 = [2 2];
% x5 = [7 7];
% y5 = [2 3];
% x6 = [7 18];
% y6 = [3 3];

%lines

% line(x1,y1,"Color","red");
% line(x2,y2,"Color","red");
% line(x3,y3,"Color","red");
% line(x4,y4,"Color","red");
% line(x5,y5,"Color","red");
% line(x6,y6,"Color","red");