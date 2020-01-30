%%% this is a programe to generate arbitrary waveform as an input for
%%% signal generator
clear;
%clc

myFGen=fgen();
getResources(myFGen)
getDrivers(myFGen)
myFGen.Resource='USB0::1689::839::C038680::0::INSTR';
%myFGen.DriverDetectionMode='manual'
%myFGen.Driver='tkafg3k'
connect(myFGen)
selectChannel(myFGen, '1');
%selectChannel(myFGen, '2');
myFGen.Waveform = 'Arb'; 

myFGen.Mode = 'continuous';




%%
%%%%% define waveform of signal
%clc
%%% base frequency: Hz
Frequency=1e1;       
%%%% time step
dt=0.001*1/Frequency;    
%%%% time over a period
t=0:dt:(1/Frequency-dt);      

%%% number of orders of harmonics
%n=2;  
%%% amplitudes
a1=1;              
a2=1;

%%% phases
phi1=0;
phi2=pi/6;

%myFGen.Amplitude = 2.0;
%%% signal for each harmonic
signal_1=a1*sin(2*pi*Frequency.*t+phi1);
signal_2=a2*sin(2*pi*2*Frequency.*t+phi2);

%%% total signal
signal=(signal_1+1.*signal_2);

%signal=signal_2;%a1*sin(2*pi*Frequency.*t);


figure(1)
plot(t,signal);



%removeWaveform(myFGen);
%removeWaveform(myFGen,10003);

d1=downloadWaveform(myFGen,signal);
enableOutput(myFGen);

removeWaveform(myFGen,10003);

%disconnect(myFGen);


%clear myFgen;