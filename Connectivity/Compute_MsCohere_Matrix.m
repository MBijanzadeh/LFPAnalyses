           
function [Coh,Coh_FreqAvg] = Compute_MsCohere_Matrix(Input,Fs) 

% This is just saving the coherence for a matrix of number of channles
% instead of 1 vector 


% Input = ERP_CAR = channel x time series 

% Coh = Chan x Chan x FreqRange

for i = 1: size(Input,1)-1  % Chan , we don't compute the coherency of each channle by itself , forming up triangle matrix 
    
    for j= i+1:size(Input,1) % 
            
            X= Input(i,:); Y = Input(j,:); 

           
            [MsCoh,Freqs] = Compute_MsCohere(X,Y,Fs);

            
            Coh(i,j,:)= MsCoh;

            
        
    end
end
    

%% Averaging the coherence for frequency of interest
   Frequency_Ranges= [4, 8,12, 30, 55, 70, 150];         
   Coh_FreqAvg= nan(size(Coh,1),size(Coh,2), numel(Frequency_Ranges)-1);  

for Freqrange = 1:numel(Frequency_Ranges)-1
    Ind = find(Freqs <=Frequency_Ranges(Freqrange+1) & Freqs>Frequency_Ranges(Freqrange)); 

    Coh_FreqAvg(:,:,Freqrange) = nanmean(Coh(:,:,Ind(1):Ind(end)),3); 
end

        
        
        
        
        
       
