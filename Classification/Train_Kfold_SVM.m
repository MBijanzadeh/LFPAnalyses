

function [SVM_FullModel,SVM_FullM_RandomLabel]=Train_Kfold_SVM(Feature,Label,Train,Test,Rand_iter) 

%% ------------------------ written by Maryam Bijanzadeh  ----------------

% Inputs:
% Feature and Label=  sample x features and sample x 1 
% Train = logical vector, is the training set coming from the partition , i.e. Partition.training 
% test = logical vector, is the test set from the partition , i.e. Partition.test
% Rand iter = scalar defining how many times the labels are shuffled for
% the random model 

% Outputs: 
% SVM_FullModel = a structure containing the trained model paramters that has Accuracy, AUC, Xlog, Ylog, FValue,
% confusion matrix and etc. 
% SVM_FullM_RandomLabel = a structure that is similar as SVM_FullModel but
% the labels are shuffled for Rand_iter times on the same trained model -->
% this is testing how the accuracy of the trained model would be if the
% labels are shuffled. 
 
PostProbs_Full = []; ConMat_Full = []; Accuracy_Full =[];FValue_SVM_Full = []; 
PostProbs_Rand = cell(1,Rand_iter); ConMat_Rand = cell(1 ,Rand_iter); Accuracy_Rand = zeros(1,Rand_iter); FValue_SVM_Rand =zeros(1,Rand_iter); 
Opt_Params = struct; 
Xlog_Rand = cell(1 ,Rand_iter); Ylog_Rand = cell(1 ,Rand_iter); Tlog_Rand = cell(1 ,Rand_iter); AUClog_Rand = zeros(1,Rand_iter); 
Xlog = []; Ylog = []; Tlog = []; AUClog=[]; 

      
    [Trained_Model,Opt_Params]= GetOptimizedFeatureSVM_KFoldV3(Feature,Label,5,Train) ; % optimizes the paramter using the training set from CVpartiotion Kfold out 
    [Accuracy_Full,ConMat_Full,FValue_SVM_Full,PostProbs_Full,NewLabels,NewLabels_Ind,Flag] = Compute_SVM_ONTest(Trained_Model,Feature,Label,Train,Test);

    
   
    % if the posterior probability is a step function retrain the model and
    % do this for 10 times at most 
    counterFlag = 1;
    while Flag && counterFlag<11

        [Trained_Model,Opt_Params]= GetOptimizedFeatureSVM_KFoldV3(Feature,Label,10,Train) ; % optimizes the paramter using Kfold out 
        [Accuracy_Full,ConMat_Full,FValue_SVM_Full,PostProbs_Full,NewLabels,NewLabels_Ind,Flag] = Compute_SVM_ONTest(Trained_Model,Feature,Label,Train,Test);
        counterFlag = counterFlag+1;
    end 
    
    [Xlog,Ylog,Tlog,AUClog] = perfcurve(Label(Test),PostProbs_Full(:,2),1); % Xlog and Ylog are useful for plotting the ROC curve, see the matlab perfcurve  
    
    
        
        
        
        
        %% Random Model , for each model shuffle labels 100 times , This random model tests whether the trained model on the data is classifying shuffled labeled with the same accuracy or not 
        
      for rept = 1:Rand_iter
            Shuff_Label = Label(randperm(size(Label,1)));
            [Ind_Train_Shuff,Ind_Test_Shuff ] = Creat_TrainTestSamples(Shuff_Label,1/(2*10)); %if you have Num_fold = 5 --> 20 % you need to pass 0.1 to the Percent ! 
          
            [Accuracy_Rand(rept),ConMat_Rand{1,rept},FValue_SVM_Rand(rept),PostProbs_Rand{1,rept}] = Compute_SVM_ONTest(Trained_Model,Feature,Shuff_Label,Ind_Train_Shuff,Ind_Test_Shuff);
            [Xlog_Rand{1,rept},Ylog_Rand{1,rept},Tlog_Rand{1,rept},AUClog_Rand(rept)] = perfcurve(Shuff_Label(Ind_Test_Shuff),PostProbs_Rand{1,rept}(:,2),1);
     
      end

%%

SVM_FullModel = struct; 
SVM_FullModel.Accuracy = Accuracy_Full; SVM_FullModel.ConMat= ConMat_Full;  SVM_FullModel.FValue = FValue_SVM_Full; SVM_FullModel.Xlog= Xlog;
SVM_FullModel.Ylog = Ylog; SVM_FullModel.AUC = AUClog;  SVM_FullModel.OptimParams = Opt_Params; SVM_FullModel.Flag = Flag; SVM_FullModel.NewLabel= NewLabels ; % the new labels comparable with abel(Test) 
SVM_FullModel.Label_Test= Label(Test) ; SVM_FullModel.Test = Test; SVM_FullModel.NewLabelTest= NewLabels_Ind; 
% NewLabels is same size as Label(Test) 
% NewLabels_Ind is just have the new labels on the total label indices 

SVM_FullM_RandomLabel = struct; 
SVM_FullM_RandomLabel.Accuracy = Accuracy_Rand; SVM_FullM_RandomLabel.ConMat= ConMat_Rand;  SVM_FullM_RandomLabel.FValue = FValue_SVM_Rand; SVM_FullM_RandomLabel.Xlog= Xlog_Rand;
SVM_FullM_RandomLabel.Ylog = Ylog_Rand; SVM_FullM_RandomLabel.AUC = AUClog_Rand;  




