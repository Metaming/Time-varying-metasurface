%%% this is a programe to generate arbitrary waveform as an input for
%%% signal generator
clear;
%clc

% close all open instruments
objs = instrfind;
fclose(objs)


%%%%%%%%%%%%%%% define signal
% frequency
Freq=1*1e3;  
% time step
dt=0.001/Freq;   
% time
t=0:dt:1/Freq-dt;   

% amplitude
a1=1; a2=0.5;
% phase
ph1=0; ph2=pi/2;
% signal components
signal_1=a1*sin(2*pi*Freq*t+ph1);
signal_2=a2*sin(2*pi*2*Freq*t+ph2);

% combined signal
signal=signal_1+signal_2;
% normalized signal to maximum amplitude 1
signal_n=1.*signal./max(abs(signal));

% off set to 0 to 1, scale it by 2^13 and take the integer part; transform it into 16-bit
% integer for input (not clear why it works)
signal_int=int16((signal_n+1)*(2^13-1)); 

figure(1);
plot(t,signal_n);


data_array = signal_int.'; 
% swap the high and low bits of digits to match the input format of the AFG
data_array = swapbytes(data_array);

%%%%%%%%%%%%%%%%%%%
% assign visa of AFG
vFG = visa('ni','GPIB0::11::INSTR');
% define buffersize for output 
vFG.outputbuffersize = 10000;

% connect to AFG
fopen(vFG);
%fprintf(vFG, '*IDN?') %%% 
%fscanf(vg)

% turn EOI mode off termporarily so command is not sent yet
vFG.EOImode = 'off';
fprintf(vFG, '%s', 'TRACE:DATA EMEMORY,#42000');
vFG.EOImode = 'on';

% write the waveform into the AFG
fwrite(vFG, data_array, 'int16');

% amplitude of the waveform
am1=2; am2=0.5;
Am1=['SOURCE1:VOLTAGE:AMPLITUDE ',num2str(am1)];
Am2=['SOURCE2:VOLTAGE:AMPLITUDE ',num2str(am2)];
fprintf(vFG,Am1);
fprintf(vFG,Am2);


fclose(vFG);
