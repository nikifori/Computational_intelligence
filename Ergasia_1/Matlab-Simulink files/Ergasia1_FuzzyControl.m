%% Nikiforidis Konstantinos 9084
% 11_Satellite
% Ergasia1_FuzzyControl.m
%%
% close all
% clear all
% Prwta trexoyme to Ergasia1_ClassicControl.m
% gia na kratithoun oi metavlites na mporoyme
% na tis xrhsimopoihsoyme kai edw


FLC_with_rules = readfis('FLC_Satellite_11_withrules');
% fuzzyLogicDesigner(FLC_with_rules);

% apo diafaneia 9 part 2 exw
Ti = Kp/Ki;
a = Ti;

% K = Kp/F{aKe} = 2.02913029039800/F{0.6764*1} = 
% 2.02913029039800/0.6764*F{1}
% Sthn arxiki fash Ke = 1
K = Kp/a;

% exodos gia E = PS kai dE/dt = ZR
output = evalfis(FLC_with_rules,[0.25 0]); % = 0.3335

% surface 
gensurf(FLC_with_rules)