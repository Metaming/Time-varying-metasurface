function spec=Fun_SA(vSA,num_points)
%%%%% this is a function to connect to spectral analyzer and read data
%clear
%fclose(vSA);

N_meas=1;
spec=zeros(num_points,N_meas);

%tic;
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
%toc
%fclose(vSA);
warning on;
end
%%
%figure(4)
%plot(spec(:,:));
