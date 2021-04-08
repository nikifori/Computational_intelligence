%% Nikiforidis Konstantinos 9084
% 11_Satellite
% Ergasia1_ClassicControl.m
%%
close all; 
clear all;

Gp_tf = zpk([],[-1 -9],10); % Gp(s)
% Arxikes times Gc_tf prin to tuning
Gc_tf = zpk(-1.5 , 0 , 1); % mideniko konta sto -1 

% tuning
% controlSystemDesigner(Gp_tf,Gc_tf);

% times Gc_tf meta to tuning
load("Gc_tf_piController.mat");
% apo to arxeio "Gc_tf_piController.mat"
Gc_tf = zpk( -1.47846592907130, 0 , 2.02913029039800);
 
System_Open_tf = series(Gc_tf,Gp_tf);
% Sinartisi kleistou vrogxou
System_Closed_tf = feedback(System_Open_tf,1);
% gia to screenshot
% step(System_Closed_tf);

% Ki kai Kp
Kp = 2.02913029039800;
c = 1.47846592907130;
Ki = c*Kp;

