function Fun_AFG(vFG,signal_1norm,signal_2norm,am1,am2,offset1,offset2)
%%% this is a function to generate arbitrary waveform for two channels of
%%% signal generator



% get normalized waveform (maximum +1 and minimum -1) from function input
signal1_n=signal_1norm;
signal2_n=signal_2norm;
% off set to 0 to 2, scale it by 2^13 and take the integer part; transform it into 16-bit
% integer for input (not clear why it works)
signal1_int=int16((signal1_n+1)*(2^13-1)); 
signal2_int=int16((signal2_n+1)*(2^13-1)); 


data_array1 = signal1_int.'; 
data_array2 = signal2_int.'; 

%figure(1);
%plot(t,data_array1,'r');hold on;
%plot(t,data_array2);hold off;

% swap the high and low bits of digits to match the input format of the AFG
data_array1 = swapbytes(data_array1);
data_array2 = swapbytes(data_array2);


%%%%%%%%%%% INPUT WAVEFORM FOR CHANNEL 1

% turn EOI mode off termporarily so command is not sent yet
vFG.EOImode = 'off';
% asign ememory for the data in AFG
fprintf(vFG, '%s', 'TRACE:DATA EMEMORY,#42000');
vFG.EOImode = 'on';
 
% write the waveform into the AFG ememory, 
fwrite(vFG, data_array1, 'int16');
% copy the waveform into the AFG ememory user1, 
fwrite(vFG,'TRACE:COPY USER1,EMEM');
% assign the waveform of user1 to channel 1
Wf1=['SOURCE1:FUNCTION USER1'];
fwrite(vFG,Wf1);

%%%%%%%%%%% INPUT WAVEFORM FOR CHANNEL 2

% turn EOI mode off termporarily so command is not sent yet
vFG.EOImode = 'off';
% asign ememory for the data in AFG
fprintf(vFG, '%s', 'TRACE:DATA EMEMORY,#42000');
vFG.EOImode = 'on';

% write the waveform into the AFG ememory, 
fwrite(vFG, data_array2, 'int16');
% copy the waveform into the AFG ememory user2, 
fwrite(vFG,'TRACE:COPY USER2,EMEM');
% assign the waveform of user1 to channel 2
Wf2=['SOURCE2:FUNCTION USER2'];
fwrite(vFG,Wf2);

%%%%%%%%%% CHANGE THE OUTPUT

% offset, amplitude and phase of the waveform
Off1=['SOURCE1:VOLTAGE:OFFSet ',num2str(offset1)];
Off2=['SOURCE2:VOLTAGE:OFFSet ',num2str(offset2)];
Am1=['SOURCE1:VOLTAGE:AMPLITUDE ',num2str(am1)];
Am2=['SOURCE2:VOLTAGE:AMPLITUDE ',num2str(am2)];
Ph1=['SOURCE1:PHASE:ADJUST ',0,'DEG'];
Ph2=['SOURCE2:PHASE:ADJUST ',90,'DEG'];

fprintf(vFG,Off1);
fprintf(vFG,Off2);
fprintf(vFG,Am1);
fprintf(vFG,Am2);
fprintf(vFG,Ph1);
fprintf(vFG,Ph2);

%fclose(vFG);
end
%
