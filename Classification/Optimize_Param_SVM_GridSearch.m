function [Opt_C,Opt_Gamma,matrixErr]= Optimize_Param_SVM_GridSearch(Label_Train,Feat_Train) 


%% This function gets the Label_Train and Feature _Train and using hold out makes the validation set 
% then runs the Grid search with different parameters of C and Gamma for
% the SVM Rbf 

vectorC = ([0.01:0.1:10]); 

vectorG = 2.^([-1:-0.25:-20]); 

P = cvpartition(Label_Train,'HoldOut',0.2); % set 20 percent aside to validating the parameters by the Grid search  
Validation = Label_Train(P.test); 
ValidationTarget = Feat_Train(P.test,:);
Train = Label_Train(P.training); 
TrainTarget =  Feat_Train(P.training,:);



[ matrixErr,GInd,CInd ] = Gridsearch_MB ( Train, TrainTarget, Validation, ValidationTarget, vectorC, vectorG );

Opt_C = vectorC(CInd); 
Opt_Gamma = vectorG(GInd); 