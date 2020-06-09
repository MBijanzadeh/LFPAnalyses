function [MsCoh,Freqs] = Compute_MsCohere(X,Y,Fs,WindowSize,NOverlap)

% ----------------- Maryam Bijanzadeh 10/22/209--------------------------
% This function computes coherence based on matlab mscoher function
% Input : X , Y , two vector matrix 
% Fs : sampling frequency 
% NOverlap = number of overlap in samples 
% we will be using hanning window with WindowSize 

if nargin <4
    WindowSize = floor(Fs/10); 
end

if nargin <5
    NOverlap = 0;  
end
 

[MsCoh,Freqs] = mscohere(X,Y,hanning(WindowSize),NOverlap,[],Fs);
    