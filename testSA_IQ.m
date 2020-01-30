

san=1;
if san==1
fwrite(vSA, 'INST SAN');
fwrite(vSA, 'CALC:UNIT:POW DBM');

fwrite(vSA, 'INIT:CONT OFF'); 
% perform sweep
fwrite(vSA, 'INIT;');
% wait for sweep to complete
fprintf(vSA, '*OPC?');
%fscanf(vSA, '%d');
fprintf(vSA, 'TRACE? TRACE1');
warning off;

line0=1;line1=0;
USB6525(sUSB,line0,line1);

read_values = Fun_SA(vSA,num_points);
spec21(:,:)=read_values.';



line0=0;line1=1;
USB6525(sUSB,line0,line1);

read_values = Fun_SA(vSA,num_points);
spec11(:,:)=read_values.';

figure(11)
subplot(1,2,1)
plot(spec21);

subplot(1,2,2)
plot(spec11);
end
%%

fwrite(vSA, 'INST IQ');
fwrite(vSA, 'INST IQ');
fwrite(vSA, 'CALC FORM RIM');
fwrite(vSA, 'CALC:UNIT:POW VOLT');
fwrite(vSA, 'DISP:TRAC:Y:MODE ABS');
%


%%
for n_test=1:1
    
fwrite(vSA, 'INIT:CONT OFF'); 

line0=1;line1=0;
USB6525(sUSB,line0,line1);
pause(0.2);
% perform sweep
tic
fwrite(vSA, 'INIT;');
% wait for sweep to complete
fprintf(vSA, '*OPC?');

fprintf(vSA, 'TRACE? TRACE1');
pause(1);

warning off;
read_values = fscanf(vSA,'%f,', 2.*num_points);

spec_real(:,:)=read_values(1:num_points);
spec_imag(:,:)=read_values(1+num_points:2*num_points);

specS21=spec_real+1i.*spec_imag;


line0=0;line1=1;
USB6525(sUSB,line0,line1);
fwrite(vSA, 'INIT:CONT OFF'); 

pause(0.2);
% perform sweep
tic
fwrite(vSA, 'INIT;');
% wait for sweep to complete
fprintf(vSA, '*OPC?');

fprintf(vSA, 'TRACE? TRACE1');
pause(1);

warning off;
read_values = fscanf(vSA,'%f,', 2.*num_points);

spec_real(:,:)=read_values(1:num_points);
spec_imag(:,:)=read_values(1+num_points:2*num_points);

toc
%
specS11=spec_real+1i.*spec_imag;

AbsspecS11=abs(specS11).^2;
specS11=sqrt(10.^((10*log10(AbsspecS11)+18.2)./10)).*exp(1i.*angle(specS11));
%
S11_test(:,n_test)=specS11;
S21_test(:,n_test)=specS21;


end;
%%
ABSS11_test=abs(S11_test);  ABSS11_ave=sum(ABSS11_test.')./n_test;
PHS11_test=angle(S11_test); PHS11_ave=sum(PHS11_test.')./n_test;
ABSS21_test=abs(S21_test);  ABSS21_ave=sum(ABSS21_test.')./n_test;
PHS21_test=angle(S21_test); PHS21_ave=sum(PHS21_test.')./n_test;
%%
figure(12)

subplot(2,1,1)
plot(abs(specS21),'k');hold on;
plot(abs(specS11),'r');hold off;
ylabel('Abs');
legend('|S21|','|S11|');

subplot(2,1,2)
plot(angle(specS21),'k');hold on;
plot(angle(specS11),'r');hold off;
ylabel('angle')
legend('arg(S21)','arg(S11)');

line0=1;line1=0;
USB6525(sUSB,line0,line1);
fwrite(vSA, 'INIT:CONT ON'); 
%%
figure(16)
subplot(4,1,1)
plot(ABSS11_ave);hold on
plot(ABSS11_ave2,'-.')
ylabel('s11')

subplot(4,1,2)
plot(ABSS21_ave);hold on
plot(ABSS21_ave2,'-.r')
ylabel('s21')

subplot(4,1,3)
plot(PHS11_ave);hold on
plot(PHS11_ave2,'-.')
ylabel('\angle s11')

subplot(4,1,4)
plot(PHS21_ave);hold on
plot(PHS21_ave2,'-.r')
ylabel('\angle s21')

%%
[cm,ci]=max(BestOBJ_allpass)
    
    a1=A1_best_allpass(ci,:);
    a2=A2_best_allpass(ci,:);
    b1=B1_best_allpass(ci,:);
    b2=B2_best_allpass(ci,:);
    ph1=PH1_best_allpass(ci,:);
    ph2=PH2_best_allpass(ci,:);
    
    am1=AM1_best_allpass(ci,:);
    am2=AM2_best_allpass(ci,:);
    offs1=OF1_best_allpass(ci,:);
    offs2=OF2_best_allpass(ci,:);
    
    
    DPH=[0:10:360].*pi/180;
    
    for ndph=1:length(DPH)
        
      dph=DPH(ndph);  
        
        
        for n=1:n_od 
        sign1_od(n,:)=a1(n).*sin(n.*(2*pi.*Freq.*t+ph1+dph))+b1(n).*cos(n.*(2*pi.*Freq.*t+ph1+dph));  
        sign2_od(n,:)=a2(n).*sin(n.*(2*pi.*Freq.*t+ph2+dph))+b2(n).*cos(n.*(2*pi.*Freq.*t+ph2+dph));
        end
                
        % total signal
        signal1=sum(sign1_od);
        signal2=sum(sign2_od);
        % normalized signal
        signal_1n=signal1./max(abs(signal1));
        signal_2n=signal2./max(abs(signal2));

        % write signal to AFG
        Fun_AFG(vFG,signal_1n,signal_2n,am1,am2,offs1,offs2);
        %fclose(vFG);

        fwrite(vSA, 'INIT:CONT OFF'); 

        line0=1;line1=0;
        USB6525(sUSB,line0,line1);
        pause(0.2);
        % perform sweep
        tic
        fwrite(vSA, 'INIT;');
        % wait for sweep to complete
        fprintf(vSA, '*OPC?');

        fprintf(vSA, 'TRACE? TRACE1');
        pause(1);

        warning off;
        read_values = fscanf(vSA,'%f,', 2.*num_points);

        spec_real(:,:)=read_values(1:num_points);
        spec_imag(:,:)=read_values(1+num_points:2*num_points);

        specS21=spec_real+1i.*spec_imag;


        line0=0;line1=1;
        USB6525(sUSB,line0,line1);
        fwrite(vSA, 'INIT:CONT OFF'); 

        pause(0.2);
        % perform sweep
        
        fwrite(vSA, 'INIT;');
        % wait for sweep to complete
        fprintf(vSA, '*OPC?');

        fprintf(vSA, 'TRACE? TRACE1');
        pause(1);

        warning off;
        read_values = fscanf(vSA,'%f,', 2.*num_points);

        spec_real(:,:)=read_values(1:num_points);
        spec_imag(:,:)=read_values(1+num_points:2*num_points);

      
        %
        specS11=spec_real+1i.*spec_imag;

        AbsspecS11=abs(specS11).^2;
        specS11=sqrt(10.^((10*log10(AbsspecS11)+18.2)./10)).*exp(1i.*angle(specS11));
        %
        S11_ph(:,ndph)=specS11;
        S21_ph(:,ndph)=specS21;
        
             


    end 
    %%
    figure(22)
    plot(angle(S21_ph(1,:)));hold on;