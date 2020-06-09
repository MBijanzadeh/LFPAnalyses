 function   [AnalyticAmp ,ComHilbert,PhaseHilbert,DsFs,BP_Fil] = ComputeComplexPower_Hilbert(Input,Fs,rejectionTimes,DsFs)
         % Compute the power first using band pass filter 2nd order
         % butterworth and then apply hilbert of matlab 
         % Phase Hilbert would be the angle of the ComHilbert 
         % initialization 
         
         % the Power is doandsample to DsFs but the Complex Hilbert is not
         % at this point and the coherency would be a downsampled version !
         % 
         FreqRange=[ 4 8 12 30 55 70 100]; 
         
         AnalyticAmp = nan(numel(FreqRange)-1,size(Input,2),size(Input,1)); % this has the analytic amplitude of the hilbert 
         PhaseHilbert = nan(numel(FreqRange)-1,size(Input,2),size(Input,1)); % This has the phase 
         BP_Fil =  nan(numel(FreqRange)-1,size(Input,2),size(Input,1)); % this has the analytic amplitude of the hilbert 
       
         ComHilbert = nan(numel(FreqRange)-1,size(Input,2),size(Input,1));% Complex Hilbert 
          
         for Freq = 1:numel(FreqRange)-1 
             BandPassFreq = [FreqRange(Freq) FreqRange(Freq+1)];
             [DD FF]=butter(4,(BandPassFreq*2)/Fs,'bandpass'); % 2nd order butterworth 
              
           for Chan = 1:size(Input,1)
   
                sprintf('Computing Hilbert for Chan :%d at frequency: %d', Chan, Freq)

             BP_Fil(Freq,:,Chan) = filtfilt(DD,FF,Input(Chan,:)); 
             XX = hilbert(filtfilt(DD,FF,Input(Chan,:))); 
             AnalyticAmp(Freq,:,Chan) = abs(XX); 
             PhaseHilbert(Freq,:,Chan) = angle(XX); 
             
             ComHilbert(Freq,:,Chan) = XX; 
          
             clear XX; 
           end 
           
         end 

         % Remove bad times from the power matrices 
 
        if ~isempty(rejectionTimes)
            for iRejection = 1:size(rejectionTimes,1)
                AnalyticAmp(:,rejectionTimes(iRejection,1):rejectionTimes(iRejection,2),:) = nan;
                ComHilbert(:,rejectionTimes(iRejection,1):rejectionTimes(iRejection,2),:) = nan;
                PhaseHilbert(:,rejectionTimes(iRejection,1):rejectionTimes(iRejection,2),:) = nan;
                BP_Fil(:,rejectionTimes(iRejection,1):rejectionTimes(iRejection,2),:) = nan;
            end

        end


        %%  downsample the power data and the Annotations to lower FS 
      

%        Power = nan(size(AnalyticAmp,1), round(size(AnalyticAmp,2)/round(Fs/DsFs)),size(AnalyticAmp,3));      
%         for Chan = 1:size(Input,1)
%             Power(:,:,Chan) = resample(AnalyticAmp(:,:,Chan)',DsFs,Fs)'; % downsample to fsDs
% 
%         end
    
      end 