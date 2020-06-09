

function [RF_Model,bestOOBErr,bestHyperparameters]= GetOptimizedModel_RandomForest(Feature,Label,Ind_Train) 

%% ------------------------ written by Maryam Bijanzadeh  9/7/2018 ----------------

% This function uses matlab built in treebagger functions to find the best parameters for RF function 
% 
% there is no need for cross validation for paramter tuning in random
% forest bc OOB error takes care of it 
% Inputs : 
% Feature = samples x feautures 
% Label = samples by 1:  0s and 1s -> binary classifier 
% 

% uses this guide : https://www.mathworks.com/help/stats/tune-random-forest-using-quantile-error-and-bayesian-optimization.html


Optimized_Params= struct; 
%% 

Label1 = Label(Ind_Train);  
Feature_Opt1 = Feature(Ind_Train,:); 


% P = cvpartition(Label1,'KFold',Num_fold);
% 
% OPT.NumGridDivisions= 15; % This is running on the default range for both e-3 - e3 
% OPT.Optimizer = 'gridsearch'; 
% OPT.CVPartition = P; 
% OPT.ShowPlots = 0; % don't show the optimization plot 
% OPT.Verbose = 0; 
% % OPT.Repartition = 1; 

maxMinLS = 20;
minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
if size(Feature_Opt1,2) ==2 % In case of PCA 
    numPTS = optimizableVariable('numPTS',[1,2],'Type','integer');
else 
    numPTS = optimizableVariable('numPTS',[1,size(Feature_Opt1,2)-1],'Type','integer');
end
hyperparametersRF = [minLS; numPTS];



results = bayesopt(@(params)oobErrRF(params,Feature_Opt1,Label1),hyperparametersRF,...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',0,'PlotFcn',[]); 

bestOOBErr = results.MinObjective;
bestHyperparameters = results.XAtMinObjective;

RF_Model = TreeBagger(300,Feature_Opt1,Label1,...
    'OOBPrediction','on','Method','classification','Surrogate','On','OOBPredictorImportance','On',...
    'MinLeafSize',bestHyperparameters.minLS,...
    'NumPredictorstoSample',bestHyperparameters.numPTS);



% 
% minLS = optimizableVariable('minLS',[1,20],'Type','integer');
% numPTS = optimizableVariable('numPTS',[1,113],'Type','integer');
% hyperparametersRF = [minLS;numPTS];
% rng(1);
% fun = @(hyp)f(hyp,X,Y);
% results = bayesopt(fun,hyperparametersRF);
% besthyperparameters = bestPoint(results);
% 
% 
% function oobMCR = f(hparams, X, Y)
% opts=statset('UseParallel',true);
% numTrees=300;
% A=TreeBagger(numTrees,X,Y,'method','classification','OOBPrediction','Surrogate','On','on','Options',opts,...
%     'MinLeafSize',hparams.minLS,'NumPredictorstoSample',hparams.numPTS);
% oobMCR = oobError(A, 'Mode','ensemble');
% end
% 
% 




%% 
function oobErr = oobErrRF(params,X,Y)
%oobErrRF Trains random forest and estimates out-of-bag quantile error
%   oobErr trains a random forest of 300 regression trees using the
%   predictor data in X and the parameter specification in params, and then
%   returns the out-of-bag quantile error based on the median. X is a table
%   and params is an array of OptimizableVariable objects corresponding to
%   the minimum leaf size and number of predictors to sample at each node.
randomForest = TreeBagger(300,X,Y,...
    'OOBPrediction','on','Method','classification','Surrogate','On','MinLeafSize',params.minLS,'OOBPredictorImportance','On',...
    'NumPredictorstoSample',params.numPTS);
oobErr = oobError(randomForest,'Mode','ensemble');
end


end

