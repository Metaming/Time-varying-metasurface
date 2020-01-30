%FREQUENCYSWEEPTEK M-Code for communicating with an instrument. 
%  
%   This is the machine generated representation of an instrument control 
%   session using a device object. The instrument control session comprises  
%   all the steps you are likely to take when communicating with your  
%   instrument. These steps are:
%       
%       1. Create a device object   
%       2. Connect to the instrument 
%       3. Configure properties 
%       4. Invoke functions 
%       5. Disconnect from the instrument 
%  
%   To run the instrument control session, type the name of the M-file,
%   frequencysweepTek, at the MATLAB command prompt.
% 
%   The M-file, FREQUENCYSWEEPTEK.M must be on your MATLAB PATH. For additional information
%   on setting your MATLAB PATH, type 'help addpath' at the MATLAB command
%   prompt.
%
%   Example:
%       frequencysweepTek;
%
%   See also ICDEVICE.
%

%   Creation time: 09-Mar-2009 15:28:00 


% Create a VISA-USB object.
interfaceObj = instrfind('Type', 'visa-usb', 'RsrcName', 'USB0::0x0699::0x0347::C038680::INSTR', 'Tag', '');

% Create the VISA-USB object if it does not exist
% otherwise use the object that was found.
if isempty(interfaceObj)
    interfaceObj = visa('tek', 'USB0::0x0699::0x0347::C038680::INSTR');
else
    fclose(interfaceObj);
    interfaceObj = interfaceObj(1);
end

% Create a device object. 
deviceObj = icdevice('tek_afg3000.mdd', interfaceObj);

% Connect device object to hardware.
connect(deviceObj);

% Configure property value(s).
set(deviceObj.Sweep(1), 'Enabled', 'on');
set(deviceObj.Sweep(1), 'Spacing', 'log');
set(deviceObj.Sweep(1), 'Start', 20.0);
set(deviceObj.Sweep(1), 'Stop', 20000.0);
set(deviceObj.Sweep(1), 'Time', 1.0);

% Disconnect device object from hardware.
disconnect(deviceObj);

% Delete objects.
delete([deviceObj interfaceObj]);
