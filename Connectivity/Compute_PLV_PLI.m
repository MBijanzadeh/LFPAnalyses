function [PLV,PLI,PLV_NoZero]= Compute_PLV_PLI(Angle) 

% Angle is the Angle of hilbert transform it is computed in
% ComputeComplexPower_Hilbert function 
% Angle is Frequnecy x Samples x Channel , this function computes the
% PLV and PLI at each time point 

% The PLV needs to be averaged across trials but since we don't have trials
% we use a moving window defined in the function that uses this subfunction


PLV = nan(size(Angle,3),size(Angle,3),size(Angle,1)); 
PLV_NoZero = nan(size(Angle,3),size(Angle,3),size(Angle,1)); 
PLI = nan(size(Angle,3),size(Angle,3),size(Angle,1)); 


for i = 1: size(Angle,3)-1  % Chan
    for j= i+1:size(Angle,3) % Chan , we don't compute the PLV/PLI of each channle by itself 
    
        for Freq=1:size(Angle,1)
            
            X= Angle(Freq,:,i); Y = Angle(Freq,:,j); 
%             sprintf('Computing Coherence of Chan :%d and %d frequency: %d', i,j, Freq)

            Delta_Angle = X- Y; 
          
            PLV(i,j,Freq) = abs(nanmean(exp(1i*Delta_Angle)));
            PLV_NoZero(i,j,Freq) = abs(imag(nanmean(exp(1i*Delta_Angle)))); % Zero lag removed it is the abs of imaginary of sum PLV 
            PLI(i,j,Freq) = abs(nanmean(sign(Delta_Angle))); 
            
            clear Delta_Angle; 
        end
    end
end
    