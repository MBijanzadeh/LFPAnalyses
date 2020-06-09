function [Partition] = Partition_Consecutive_Time(Label,Fold_Num) 


%%--------------Written by Maryam Bijanzadeh 6/11/2018---------------------
% This function generates similar to Kfold but the test sets in time 
% doesn't overlap with train test,like the frist 10 percent is selected 
% for the test in fold 1 and the remainder is train
% there will be no validation set This will be used by Macro_Kfold_SVM or
% any other classifier 

% It is totllay anticausal , here we don't care if the test set is
% preceeding the train set, we just care the samples are not randomly
% selected to have same features. Although still the end of test set and
% the begining of train set might share a feature since they are back to
% back but not all of it 


fold_size= floor( numel(Label)/Fold_Num); 
fold_size_EachLabel= round(fold_size/2); 

Label_One_Ind = find(Label>0.7); 
Label_Zero_Ind = find(Label<0.5); 

Partition = struct; 
% Partition.training = struct; 
% Partition.test = struct; 
% 
for K=1:Fold_Num   % It is totllay anticausal 
    
   
   Partition.training{K} = zeros(numel(Label),1); % This is in the same format as matlab partition to make the usage easier     
   Partition.test{K} = zeros(numel(Label),1); 
 
   if K==Fold_Num % If there is any extrac lables so the number of labels is not devidable by the fold number the extra ones would be include in the last fold 
      
         Partition.test{K}(Label_One_Ind(fold_size_EachLabel*(K-1)+1:end)) = Label(Label_One_Ind(fold_size_EachLabel*(K-1)+1:end));   
         Partition.test{K}(Label_Zero_Ind(fold_size_EachLabel*(K-1)+1:end)) = ones(numel(Label_Zero_Ind(fold_size_EachLabel*(K-1)+1:end)),1);   
        
   else 
       
   Partition.test{K}(Label_One_Ind(fold_size_EachLabel*(K-1)+1:fold_size_EachLabel*K)) = Label(Label_One_Ind(fold_size_EachLabel*(K-1)+1:fold_size_EachLabel*K));   
   Partition.test{K}(Label_Zero_Ind(fold_size_EachLabel*(K-1)+1:fold_size_EachLabel*K)) = ones(numel(fold_size_EachLabel*(K-1)+1:fold_size_EachLabel*K),1);   
    
   end
    
   Partition.training{K}(setdiff([1:numel(Label)],find(Partition.test{K}))) =ones((numel(Label)-numel(find(Partition.test{K}))),1); 

    
    
end 




