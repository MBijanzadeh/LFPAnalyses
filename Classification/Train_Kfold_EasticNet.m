

function [Model]=Train_Kfold_EasticNet(Feature,Label,Train,Test) 

%% ------------------------ written by Maryam Bijanzadeh 10/12/2018 ----------------

%  This function trains a binary classifier using lassoglm from matlab ,
%  there is no optimization for the alpha ration between L1 and L2 methods
%  and the alpha is set to 0.5 : equal contribution from both
%  regularization methods 

% Inputs:
% Feature and Label=  sample x features and sample x 1 
% Train = logical vector, is the training set coming from the partition , i.e. Partition.training 
% test = logical vector, is the test set from the partition , i.e. Partition.test
% Rand iter = scalar defining how many times the labels are shuffled for
% the random model 

% Outputs: 
% Model = a structure containing the trained model paramters that has Accuracy, AUC, Xlog, Ylog, FValue,
% confusion matrix and etc. 
% RF_RandomModel = a structure that is similar as Model but
% the labels are shuffled for Rand_iter times on the same trained model -->
% this is testing how the accuracy of the trained model would be if the
% labels are shuffled. 
 

[Beta,FitInfo] = lassoglm(Feature(Train,:),Label(Train),'binomial','CV',5,'alpha',0.5);
idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
B0 = FitInfo.Intercept(idxLambdaMinDeviance);
coef = [B0; Beta(:,idxLambdaMinDeviance)]; 


yhat = glmval(coef,Feature(Test,:),'logit');
yhatBinom = (yhat>=0.5);

[ConMat,order] = confusionmat(Label(Test),double(yhatBinom)); 
Accuracy = (ConMat(1,1)+ConMat(2,2))/sum(reshape(ConMat,[],1));    

FValue = compute_FValue_ConfusionMat(ConMat);  


%%

Model = struct; 
Model.Accuracy = Accuracy; Model.ConMat= ConMat;  Model.FValue = FValue; Model.FitInfo= FitInfo; Model.Beta = coef; 



