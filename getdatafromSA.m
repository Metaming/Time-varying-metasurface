%%%%% this is a program to connect to spectral analyzer and read data
%clear
objs = instrfind;
fclose(objs)
%fclose(vSA);
vSA = visa('ni','GPIB0::21::INSTR');
vSA.inputbuffersize = 10000;
fopen(vSA);

% Show display window
fwrite(vSA, 'SYST:DISP:UPD ON');
% define central frequency
fwrite(vSA, 'FREQ:CENT 3.3 GHz');
% define frequency span
fwrite(vSA, 'FREQ:SPAN 30 MHz');
% define rf bandwidth
fwrite(vSA, 'BAND 20 kHz');
% define video bandwidth
fwrite(vSA, 'BAND:VIDEO 50 kHz');
% number of sweep point
num_points = 301;
fprintf(vSA, 'SWEEP:POINTS %d\n', num_points);

% define data type as ASCII
fwrite(vSA, 'FORM ASCII');

N_meas=1;
spec=zeros(num_points,N_meas);

tic;
for n_meas=1:1:N_meas
    
% contintous (ON)/single sweep (OFF)
fwrite(vSA, 'INIT:CONT OFF'); 
% perform sweep
fwrite(vSA, 'INIT;');

% wait for sweep to complete
fprintf(vSA, '*OPC?');
%fscanf(vSA, '%d');


fprintf(vSA, 'TRACE? TRACE1');
warning off;
read_values = fscanf(vSA, '%f,', num_points);

spec(:,n_meas)=read_values;

end
toc
%fclose(vSA);
warning on;

%%
figure(4)
plot(spec(:,:));
