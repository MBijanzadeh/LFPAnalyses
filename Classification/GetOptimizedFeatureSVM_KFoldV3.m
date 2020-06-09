

function [Trained_Model,Optimized_Params,Feature_Out,Label_Out]= GetOptimizedFeatureSVM_KFoldV3(Feature,Label,Num_fold,Ind_Train) 

%% ------------------------ written by Maryam Bijanzadeh ----------------

% This function uses matlab built in SVM functions to find the best parameters for RBF function 
% gamma and C on of a given SVM you can call Trained_Model to test on the
% Ind_Test data 

% Inputs : 
% Feature = samples x feautures 
% Label = samples by 1:  0s and 1s -> binary classifier 
% Num_fold = it puts 1/Num_fold for the test and then again uses K-fold
% for SVM cross validation to asses the best parameters 


% Optimized_Params is a structure containing optimized gamma and C from the grid search
% TestSet = are the indices of the Label vector that are saved for further test for each
% combination ! 


Optimized_Params= struct; 
%% 

Label1 = Label(Ind_Train);  
Feature_Opt1 = Feature(Ind_Train,:); 


P = cvpartition(Label1,'KFold',Num_fold);

OPT.NumGridDivisions= 15; % This is running on the default range for both e-3 - e3 
OPT.Optimizer = 'gridsearch'; 
OPT.CVPartition = P; 
OPT.ShowPlots = 0; % don't show the optimization plot 
OPT.Verbose = 0; 
% OPT.Repartition = 1; 

% Get the optimized paramater by Num_fold cross validating the training set
% 
Mdl = fitcsvm(Feature_Opt1,Label1,'Solver','SMO','KernelFunction','rbf','OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',OPT,'Prior','uniform'); 


close all

% Retrain the model with the optmized parameters on the validation set 
Optimized_Params.Gamma = Mdl.KernelParameters.Scale; 
Optimized_Params.BoxConstraint = Mdl.BoxConstraints(end); 

Trained_Model = fitcsvm(Feature_Opt1,Label1,'KernelFunction','rbf','Solver','SMO','KernelScale',Optimized_Params.Gamma,'BoxConstraint',...
    Optimized_Params.BoxConstraint,'Prior','uniform'); 




