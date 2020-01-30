%%% this is a program that use simulated scattering parameters of a periodic array of metaunits in a parallel plate waveguide to generate sideband signals by mimiking a time-varying modulation. Genetic algorithm is used to optimze the modulation signal.

clear;
clc
       ep0 = 8.854418e-15; %%%%%%%% unit: C/(V*mm)
        u0 = 4*pi*1e-10; %%%%%%%% unit: V*s/(A*mm)
        c=(ep0*u0)^(-0.5);  %%%%%% unit: mm/s
        
        

s21fin1=fopen('s21_ESRR+SRRin1_period_px10py15.5_RO3.54.txt');
s21phfin1=fopen('s21phase_ESRR+SRRin1_period_px10py15.5_RO3.54.txt');
s11fin1=fopen('s11_ESRR+SRRin1_period_px10py15.5_RO3.54.txt');
s11phfin1=fopen('s11phase_ESRR+SRRin1_period_px10py15.5_RO3.54.txt');


s21fout1=fopen('s21_out1.txt','w');
s21phfout1=fopen('s21phase_out1.txt','w');
s11fout1=fopen('s11_out1.txt','w');
s11phfout1=fopen('s11phase_out1.txt','w');



while ~feof(s21fin1) 
    line1 = fgetl(s21fin1);
    line2 = fgetl(s11fin1);
    line3 = fgetl(s21phfin1);
    line4 = fgetl(s11phfin1);
    
 if ~isempty(line1)
  if (line1(9)== 'F')||(line1(1)== '-');
        
        status=fseek(s21fin1,0,0);
        status=fseek(s11fin1,0,0);
        status=fseek(s21phfin1,0,0);
        status=fseek(s11phfin1,0,0);
        
        
  else fprintf(s21fout1,'%s\r\n', line1);
       fprintf(s11fout1,'%s\r\n', line2);
       fprintf(s21phfout1,'%s\r\n', line3);
       fprintf(s11phfout1,'%s\r\n', line4);
 
    
  end
 else continue
  end
    %end
       
end


fclose('all');
s21f=load('s21_out1.txt');
s21phf=load('s21phase_out1.txt');
s11f=load('s11_out1.txt');
s11phf=load('s11phase_out1.txt');
%%
npar1=55;npar=npar1*npar1;

F=s21f(1:1001,1);
s21=reshape(s21f(:,2),1001,npar);
s21ph=reshape(s21phf(:,2),1001,npar);
s11=reshape(s11f(:,2),1001,npar);
s11ph=reshape(s11phf(:,2),1001,npar);


s21n=zeros(1001,npar);
s21phn=zeros(1001,npar);
s11n=zeros(1001,npar);
s11phn=zeros(1001,npar);

sq=[1:1:npar];
s21n(:,1:npar)=s21(:,sq);
s21phn(:,1:npar)=s21ph(:,sq);
s11n(:,1:npar)=s11(:,sq);
s11phn(:,1:npar)=s11ph(:,sq);

s21_cc(:,:,:)=reshape(s21n,1001,npar1,npar1);
s21ph_cc(:,:,:)=reshape(s21phn,1001,npar1,npar1);
s11_cc(:,:,:)=reshape(s11n,1001,npar1,npar1);
s11ph_cc(:,:,:)=reshape(s11phn,1001,npar1,npar1);




V1=-0.8:0.2:10;
V2=-0.8:0.2:10;
V1I=-0.8:0.02:10;
V2I=-0.8:0.02:10;

% here we flip the phase from CST simulation so that it follows the convention of exp(-iwt)
S21_cc=s21_cc.*exp(-1i.*s21ph_cc.*pi/180);
S11_cc=s11_cc.*exp(-1i.*s11ph_cc.*pi/180);
   
%%   
ff=4.00;
[dd,fi]=min(abs(F-ff));
    
   
    Rs21c(:,:)=real(S21_cc(fi,:,:));
    Is21c(:,:)=imag(S21_cc(fi,:,:));
    Rs11c(:,:)=real(S11_cc(fi,:,:));
    Is11c(:,:)=imag(S11_cc(fi,:,:));
    
    s21_C(:,:)=S21_cc(fi,:,:);
    s11_C(:,:)=S11_cc(fi,:,:);
    
    
    
    for j=1:length(V2I)
        
        for k=1:length(V1)
            Rs21c_I(k,:)=interp1(V2,Rs21c(k,:),V2I,'spline');
            Is21c_I(k,:)=interp1(V2,Is21c(k,:),V2I,'spline');
            Rs11c_I(k,:)=interp1(V2,Rs11c(k,:),V2I,'spline');
            Is11c_I(k,:)=interp1(V2,Is11c(k,:),V2I,'spline');
        end
        
        Rs21_CI(:,j)=interp1(V1,Rs21c_I(:,j),V1I,'spline');
        Is21_CI(:,j)=interp1(V1,Is21c_I(:,j),V1I,'spline');
        Rs11_CI(:,j)=interp1(V1,Rs11c_I(:,j),V1I,'spline');
        Is11_CI(:,j)=interp1(V1,Is11c_I(:,j),V1I,'spline');
    end
 
    
     
  %
 dz=0;
 s11_CI=(Rs11_CI+1i.*Is11_CI).*exp(1i.*ff.*1e9*2*pi/c*dz);
 s21_CI=(Rs21_CI+1i.*Is21_CI).*exp(1i.*ff.*1e9*2*pi/c*dz); %%% forward scattering needs substract incident wave
    
 %
figure(190)
subplot(2,2,1)
sf=contour(V1I,V2I,abs(s11_CI),25);
%set(sf, 'EdgeColor', 'none');
xlabel('V1(V)');
ylabel('V2(V)');
axis([min(V1),max(V1),min(V2),max(V2)])
zlabel('s11');
view([0,90])
tt=horzcat('|S11|, ','Freq=',num2str(ff),' GHz'); 
colorbar;hold off;
title(tt)

subplot(2,2,2)
sf=contour(V1I,V2I,abs(s21_CI),25);
%set(sf, 'EdgeColor', 'none');
xlabel('V1(V)');
ylabel('V2(V)');
axis([min(V1),max(V1),min(V2),max(V2)])
zlabel('s21');
view([0,90])
tt=horzcat('|S21|, ','Freq=',num2str(ff),' GHz'); 
colorbar;hold off 
title(tt)
 
 
subplot(2,2,3)
sf=mesh(V1I,V2I,angle(s11_CI));
%set(sf, 'EdgeColor', 'none');
xlabel('V1(V)');
ylabel('V2(V)');
axis([min(V1),max(V1),min(V2),max(V2)])
zlabel('s11');
view([0,90])
tt=horzcat('arg(S11), ','Freq=',num2str(ff),' GHz'); 
colorbar;hold off;
title(tt) 

subplot(2,2,4)
sf=mesh(V1I,V2I,angle(s21_CI));
%set(sf, 'EdgeColor', 'none');
xlabel('V1(V)');
ylabel('V2(V)');
axis([min(V1),max(V1),min(V2),max(V2)])
zlabel('s11');
view([0,90])
tt=horzcat('arg(S21), ','Freq=',num2str(ff),' GHz'); 
colorbar;hold off;
title(tt) 

%%
figure(112)
subplot(2,2,1)
sf=contour(V1,V2,abs(s11_C),25);
%set(sf, 'EdgeColor', 'none');
xlabel('V1(V)');
ylabel('V2(V)');
axis([min(V1),max(V1),min(V2),max(V2)])
zlabel('s11');
view([0,90])
tt=horzcat('|S11|, ','Freq=',num2str(ff),' GHz'); 
colorbar;hold off;
title(tt)

subplot(2,2,2)
sf=contour(V1,V2,abs(s21_C),25);
%set(sf, 'EdgeColor', 'none');
xlabel('V1(V)');
ylabel('V2(V)');
axis([min(V1),max(V1),min(V2),max(V2)])
zlabel('s21');
view([0,90])
tt=horzcat('|S21|, ','Freq=',num2str(ff),' GHz'); 
colorbar;hold off 
title(tt)
 
 
subplot(2,2,3)
sf=mesh(V1,V2,angle(s11_C));
%set(sf, 'EdgeColor', 'none');
xlabel('V1(V)');
ylabel('V2(V)');
axis([min(V1),max(V1),min(V2),max(V2)])
zlabel('s11');
view([0,90])
tt=horzcat('arg(S11), ','Freq=',num2str(ff),' GHz'); 
colorbar;hold off;
title(tt) 

subplot(2,2,4)
sf=mesh(V1,V2,angle(s21_C));
%set(sf, 'EdgeColor', 'none');
xlabel('V1(V)');
ylabel('V2(V)');
axis([min(V1),max(V1),min(V2),max(V2)])
zlabel('s11');
view([0,90])
tt=horzcat('arg(S21), ','Freq=',num2str(ff),' GHz'); 
colorbar;hold off;
title(tt) 

 %%  
 %%%%% remove the additional phase accumulated along the waveguide
 dz=0;
 s11_CI=(Rs11_CI+1i.*Is11_CI).*exp(1i.*ff.*1e9*2*pi/c*dz);
 s21_CI=(Rs21_CI+1i.*Is21_CI).*exp(1i.*ff.*1e9*2*pi/c*dz); %%% forward scattering needs substract incident wave
 %%s21_CI=1-s11_CI;
 

Om=0.5e6;    %%%% Hz, frequency of modulation
t=-1.0/Om:1e-3/Om:1.0/Om;
 
phi1=0.*pi/180;  %%%% relative phase of capaciance modulation 
phi2=0.*pi/180;
 



%%
AM=2;

Om=0.5e6;    %%%% Hz, frequency of modulation
t=-1.0/Om:1e-3/Om:1.0/Om;
V1_0=2;  %%% center of biased voltage
V2_0=2;

deltaV1=2;deltaV2=2; %%% amplitude of biased voltage

V1_t=V1_0+deltaV1.*cos(2.*pi.*Om.*t+phi1);
V2_t=V2_0+deltaV2.*cos(2.*pi.*Om.*t+phi2);    %
 

phi1=0.*pi/180;  %%%% relative phase of capaciance modulation 
phi2=0.*pi/180;


a1=1;a2=0.0;a3=0.0;

for na=1:length(AM)
    
    am=AM(na);
    
dc1=0.063*am;   %%%% amplitude of capaciance modulation 
dc2=0.03*am;

Omg=2.*pi.*Om;
V1_t=V1_0+deltaV1.*(a1*cos(Omg.*t+phi1)+a2*cos(2*Omg.*t+2*phi1)+a3*cos(3*Omg.*t+3*phi1)); 
V2_t=V2_0+deltaV2.*(a1*cos(Omg.*t+phi2)+a2*cos(2*Omg.*t+2*phi2)+a3*cos(3*Omg.*t+3*phi2));


for nt=1:length(t)
    
[V1a,nV1_a]=min(abs(V1_t(nt)-V1I));
[V2a,nV2_a]=min(abs(V2_t(nt)-V2I));

Rs11_t(nt)=real(s11_CI(nV2_a,nV1_a));
Rs21_t(nt)=real(s21_CI(nV2_a,nV1_a));
Is11_t(nt)=imag(s11_CI(nV2_a,nV1_a));
Is21_t(nt)=imag(s21_CI(nV2_a,nV1_a));

V1IT(nt)=V1I(nV1_a);
V2IT(nt)=V2I(nV2_a); 
end



[pl,rs]=polyfit(t,Rs11_t,40);
[Rs11_tfit,delta]=polyval(pl,t,rs);

[pl,rs]=polyfit(t,Rs21_t,40);
[Rs21_tfit,delta]=polyval(pl,t,rs);

[pl,rs]=polyfit(t,Is11_t,40);
[Is11_tfit,delta]=polyval(pl,t,rs);

[pl,rs]=polyfit(t,Is21_t,40);
[Is21_tfit,delta]=polyval(pl,t,rs);


t0=-0.5/Om:1e-3/Om:0.5/Om-1e-3/Om;




s11_t=Rs11_t(501:1500)+1i.*Is11_t(501:1500);
s21_t=Rs21_t(501:1500)+1i.*Is21_t(501:1500);

s11_t=abs(s11_t).*exp(1i.*angle(s11_t));
s21_t=abs(s21_t).*exp(1i.*angle(s21_t));



%%% calculate the fourier spectrum

%%
t_ssb=exp(1i.*2*pi.*Om.*t0);% t_ssb is to test single side band
t_ssb=exp(1i.*(pi.*sin(2*pi*Om.*t0)));% t_ssb is to test single side band
nmax=8;

for n=0:1:nmax

fa=cos(2*pi.*Om.*n.*t0);
fb=sin(2*pi.*Om.*n.*t0);
dx=2*pi*Om.*(t0(2)-t0(1));

 
a_rf(n+1)=1/pi.*sum(s11_t.*fa)*dx;
b_rf(n+1)=1/pi.*sum(s11_t.*fb)*dx;
a_tr(n+1)=1/pi.*sum(s21_t.*fa)*dx;
b_tr(n+1)=1/pi.*sum(s21_t.*fb)*dx;

a_tr_ssb(n+1)=1/pi.*sum(t_ssb.*fa)*dx;
b_tr_ssb(n+1)=1/pi.*sum(t_ssb.*fb)*dx;

end

% note that for positive order, it means exp(-i\Omega t)=cos - i sin
s11_p=0.5*(a_rf-1i.*b_rf);
s11_n=0.5*(a_rf+1i.*b_rf);
s21_p=0.5*(a_tr-1i.*b_tr);
s21_n=0.5*(a_tr+1i.*b_tr);

s21_p_ssb=0.5*(a_tr_ssb-1i.*b_tr_ssb);
s21_n_ssb=0.5*(a_tr_ssb+1i.*b_tr_ssb);




As11=[fliplr(s11_p(2:end)),s11_n];
As21=[fliplr(s21_p(2:end)),s21_n];

As21_ssb=[fliplr(s21_p_ssb(2:end)),s21_n_ssb];

%AS11_AM(na,:)=As11;
%AS21_AM(na,:)=As21;
end
%%

N=-nmax:1:nmax;

figure(101)
%subplot(1,3,1)
stem(-nmax:1:nmax,abs(As21_ssb));hold on;
%stem(-nmax:1:nmax,abs(As21_ssb2));hold off;
%plot(AM,abs(AS21_AM(:,7:end)).^2);hold off;
figure(100)
subplot(1,3,1)
plot(t0,V1_t(501:1500));hold on
plot(t0,V2_t(501:1500),'r');hold off
legend('v1','v2')
subplot(1,3,2)
plot(t0,abs(s11_t));hold on;
plot(t0,abs(s21_t));hold off;
legend('s11','s21')
subplot(1,3,3)
plot(t0,phase(s11_t));hold on;
plot(t0,phase(s21_t));hold off;
legend('arg s11','arg s21')

%%%%%%%%%%%%%%%%%% voltage dependent capacitance

 

figure(200)
subplot(1,3,1)
sf=contour(V1I,V2I,abs(s11_CI),25);
%set(sf, 'EdgeColor', 'none');
xlabel('V1(V)');
ylabel('V2(V)');
axis([min(V1),max(V1),min(V2),max(V2)])
zlabel('s11');
view([0,90])
tt=horzcat('|S11|, ','Freq=',num2str(ff),' GHz'); 
colorbar;hold on;
plot(V1_t,V2_t,'r');hold off
title(tt)

subplot(1,3,2)
sf=contour(V1I,V2I,abs(s21_CI),25);
%set(sf, 'EdgeColor', 'none');
xlabel('V1(V)');
ylabel('V2(V)');
axis([min(V1),max(V1),min(V2),max(V2)])
zlabel('s21');
view([0,90])
tt=horzcat('|S21|, ','Freq=',num2str(ff),' GHz'); 
colorbar;hold on 
title(tt)
plot(V1_t,V2_t,'r');hold off

subplot(1,3,3)
plot(t,V1_t);hold on;
plot(t,V2_t,'-.');hold off;
xlabel('t(s)');
ylabel('V1,V2(V)');
legend('V1','V2')
colorbar;hold on;



figure(201)
subplot(1,3,1)
sf=contour(V1I,V2I,angle(s11_CI),25);
%set(sf, 'EdgeColor', 'none');
xlabel('V1(V)');
ylabel('V2(V)');
axis([min(V1),max(V1),min(V2),max(V2)])
zlabel('s11');
view([0,90])
tt=horzcat('arg(S11), ','Freq=',num2str(ff),' GHz'); 
colorbar;hold on;
plot(V1_t,V2_t,'r');hold off
title(tt)

subplot(1,3,2)
sf=contour(V1I,V2I,angle(s21_CI),25);
%set(sf, 'EdgeColor', 'none');
xlabel('V1(V)');
ylabel('V2(V)');
axis([min(V1),max(V1),min(V2),max(V2)])
zlabel('s11');
view([0,90])
tt=horzcat('arg(S21), ','Freq=',num2str(ff),' GHz'); 
colorbar;hold on;
plot(V1_t,V2_t,'r');hold off
title(tt)

subplot(1,3,3)
plot(t,V1_t);hold on;
plot(t,V2_t,'-.');hold off;
xlabel('t(s)');
ylabel('V1,V2 ');
legend('V1','V2')
colorbar;hold on;
%% optimization

warning on;
optimiz=1;
fit=0;

 
% relative phase of capaciance modulation
phi1=0.*pi/180;   phi2=0.*pi/180;

% voltage dependent capacitance

Vf=1.8; 
V1_0=2;
V2_0=2;
deltaV1=1.5;deltaV2=1.5;

V1_t=V1_0+deltaV1.*cos(2.*pi.*Om.*t+phi1)/2;
V2_t=V2_0+deltaV2.*cos(2.*pi.*Om.*t+phi2)/2;
 

% initial input parameters of waveform
% amplitude of modulation (offset tuned manually)
am1=deltaV1; am2=deltaV2;
% relative phase
phi1=0.*pi/180;   
phi2=0.*pi/180;
% offset
offs1=V1_0;offs2=V2_0;


%%% degree of mutation in bias offset, amplitude and phases

% mutation in waveform coefficients
mutco=0.05; 
% mutation in relative phase

mutph=5*pi/180; 
% mutation in bias offset
mutof=0.10;
% mutation in voltage amplitude
mutam=0.10;

% when flag_sym=0, the sidebands coefficients have central symmetry
flag_sym=1;
% when iden_waveform=1, v1 and v2 have the same waveform
iden_waveform=0;

n_od=8;  %%% number of orders in modulation
N_GR=50; %%% number of generation
N_Geno=50; %%% number of genos
nmax=15;     %%%% number of orders of harmonics for HB analysis
n_objhm=nmax;

n_round=50;  %%% number of rounds of optimization


BestOBJ_allpass=zeros(n_round,1);
AS21_best_allpass=zeros(n_round,nmax*2+1); 
AS11_best_allpass=zeros(n_round,nmax*2+1); 

A_best_allpass=zeros(n_round,2*n_od-1); 

BestOBJ_allpass=zeros(n_round,1);

AS21_best_allpass=zeros(n_round,nmax*2+1); 
AS11_best_allpass=zeros(n_round,nmax*2+1); 

A1_best_allpass=zeros(n_round,n_od); 
A2_best_allpass=zeros(n_round,n_od); 
B1_best_allpass=zeros(n_round,n_od); 
B2_best_allpass=zeros(n_round,n_od); 
PH1_best_allpass=zeros(n_round,1); 
PH2_best_allpass=zeros(n_round,1); 
AM1_best_allpass=zeros(n_round,1); 
AM2_best_allpass=zeros(n_round,1); 
OF1_best_allpass=zeros(n_round,1); 
OF2_best_allpass=zeros(n_round,1); 

%

for pn=1:n_round
    
    %%%% NA: new amplitude for different orders of modulation for different
        %%%% genos, it is generated by hybridiing the best 4 and sorting the
        %%%% other genos based on performance

    NA1=zeros(N_Geno,n_od);NA2=zeros(N_Geno,n_od);
    NB1=zeros(N_Geno,n_od);NB2=zeros(N_Geno,n_od);
    NPH1=zeros(N_Geno,1);NPH2=zeros(N_Geno,1);
    NAM1=zeros(N_Geno,1);NAM2=zeros(N_Geno,1);
    NOF1=zeros(N_Geno,1);NOF2=zeros(N_Geno,1);
    %%%% the matrix of sorted different orders of amplitude of modulation for different genos
    %%%% according to the objective function
    SortA1=zeros(n_od,N_Geno);SortA2=zeros(n_od,N_Geno);
    SortB1=zeros(n_od,N_Geno);SortB2=zeros(n_od,N_Geno);
    SortPH1=zeros(1,N_Geno);SortPH2=zeros(1,N_Geno);
    SortAM1=zeros(1,N_Geno);SortAM2=zeros(1,N_Geno);
    SortOF1=zeros(1,N_Geno);SortOF2=zeros(1,N_Geno);
    %%% the objective function 
    BestOBJ=zeros(1,N_GR);

    %%% the S parameters of different orders of harmonics for different genos
    As21_Gn=zeros(N_Geno,nmax*2+1);
    As11_Gn=zeros(N_Geno,nmax*2+1);

    %%% S parameters for the best geno in each generation
    Best_As21=zeros(N_GR,nmax*2+1);
    Best_As11=zeros(N_GR,nmax*2+1);

    %%% amplitude and phase for the best geno in each generation
    Best_A1=zeros(n_od,N_GR);
    Best_A2=zeros(n_od,N_GR);

    Best_B1=zeros(n_od,N_GR);
    Best_B2=zeros(n_od,N_GR);

    Best_PH1=zeros(1,N_GR);
    Best_PH2=zeros(1,N_GR);

    Best_AM1=zeros(1,N_GR);
    Best_AM2=zeros(1,N_GR);
    
    Best_OF1=zeros(1,N_GR);
    Best_OF2=zeros(1,N_GR);

    %%% amplitude of modulation harmonics
    A1=zeros(N_GR,n_od);B1=zeros(N_GR,n_od);PH1=zeros(N_GR,1);AM1=zeros(N_GR,1);OF1=zeros(N_GR,1);
    A2=zeros(N_GR,n_od);B2=zeros(N_GR,n_od);PH2=zeros(N_GR,1);AM2=zeros(N_GR,1);OF2=zeros(N_GR,1);
    
    %%% best four genos
    BestA1=zeros(4,n_od);BestA2=zeros(4,n_od);
    BestB1=zeros(4,n_od);BestB2=zeros(4,n_od);
    BestPH1=zeros(4,1);BestPH2=zeros(4,1);
    BestAM1=zeros(4,1);BestAM2=zeros(4,1);
    BestOF1=zeros(4,1);BestOF2=zeros(4,1);
    
    sign1_od=zeros(n_od,length(t));
    sign2_od=zeros(n_od,length(t));
    %%% define the coefficient of input waveform
    % for channel 1
    a1_0=[1,random('unif',-mutco,mutco,1,n_od-1)];  %%%% initial amplitude for different orders
    b1_0=[1,random('unif',-mutco,mutco,1,n_od-1)];  %%%% initial amplitude for different orders
    ph1_0=phi1+0.*random('unif',-mutph,mutph,1,1);  %%%% initial phase for different orders
    % for channel 2
    a2_0=[1,random('unif',-mutco,mutco,1,n_od-1)];  %%%% initial amplitude for different orders
    b2_0=[1,random('unif',-mutco,mutco,1,n_od-1)];  %%%% initial amplitude for different orders
    ph2_0=phi2+random('unif',-mutph,mutph,1,1);  %%%% initial phase for different orders

    am1_0=am1.*(1+random('unif',-mutam,mutam,1,1));
    am2_0=am2.*(1+random('unif',-mutam,mutam,1,1));
    offs1_0=offs1.*(1+random('unif',-mutof,mutof,1,1))+random('unif',-mutof,mutof,1,1);
    offs2_0=offs2.*(1+random('unif',-mutof,mutof,1,1))+random('unif',-mutof,mutof,1,1);
                         
    
    
                        if am1_0+offs1>max(V1) am1_0=max(V1)-offs1;
                        end
                        if -am1_0+offs1<min(V1) am1_0=offs1-min(V1);
                        end
                        if am2_0+offs2>max(V2) am2_0=max(V2)-offs2;
                        end
                        if -am2_0+offs2<min(V2) am2_0=offs2-min(V2);                            
                        end
   
        %%%%% Generation
        for n_gr=1:1:N_GR

             %%%%% Geno
                for n_geno=1:1:N_Geno


                      %%% for the second generation use new genos formed based on the last
                      %%% generation
                    if n_gr>1  

                        a1_0=NA1(n_geno,:); a2_0=NA2(n_geno,:);
                        b1_0=NB1(n_geno,:); b2_0=NB2(n_geno,:); 
                        ph1_0=NPH1(n_geno,:); ph2_0=NPH2(n_geno,:); 
                        am1_0=NAM1(n_geno,:);am2_0=NAM2(n_geno,:);
                        offs1_0=NOF1(n_geno,:);offs2_0=NOF2(n_geno,:);

                    end


                  %%% for geno number larger than 5, introduce mutation
                      if n_geno>4
                        a1=a1_0+random('unif',-mutco,mutco,1,n_od);
                        a2=a2_0+random('unif',-mutco,mutco,1,n_od);
                        b1=b1_0+random('unif',-mutco,mutco,1,n_od);
                        b2=b2_0+random('unif',-mutco,mutco,1,n_od);

                        ph1=ph1_0+0.*random('unif',-mutph,mutph,1,1);
                        ph2=ph2_0+random('unif',-mutph,mutph,1,1);

                        am1=am1_0.*(1+random('unif',-mutam,mutam,1,1));
                        am2=am2_0.*(1+random('unif',-mutam,mutam,1,1));
                        
                        offs1=offs1_0.*(1+random('unif',-mutof,mutof,1,1))+random('unif',-mutof,mutof,1,1);
                        offs2=offs2_0.*(1+random('unif',-mutof,mutof,1,1))+random('unif',-mutof,mutof,1,1);
     
                        
                        % limit the amplitude and offset within the bias
                        % voltage
                        if am1+offs1>max(V1) am1=max(V1)-offs1;
                        end
                        if -am1+offs1<min(V1) am1=offs1-min(V1);
                        end
                        if am2+offs2>max(V2) am2=max(V2)-offs2;
                        end
                        if -am2+offs2<min(V2) am2=offs2-min(V2);                            
                        end
                      % for geno number <4, use the hybrid geno without mutation
                      else
                        a1=a1_0;a2=a2_0;
                        b1=b1_0;b2=b2_0;
                        ph1=ph1_0;ph2=ph2_0;
                        am1=am1_0;am2=am2_0;
                        offs1=offs1_0;offs2=offs2_0;
                      end
                      
                      
                      
                % define the spectral component of modulation
                for n=1:n_od 
                sign1_od(n,:)=flag_sym.*a1(n).*sin(n.*(2*pi.*Om.*t+ph1))+b1(n).*cos(n.*(2*pi.*Om.*t+ph1));  
                
                if iden_waveform==1
                    
                    a2(n)=a1(n);b2(n)=b1(n);ph2=ph2;
                    
                end
                
                
                sign2_od(n,:)=flag_sym.*a2(n).*sin(n.*(2*pi.*Om.*t+ph2))+b2(n).*cos(n.*(2*pi.*Om.*t+ph2));
                
                
                
                end
                
                % total signal
                signal1=sum(sign1_od);
                signal2=sum(sign2_od);
                % normalized signal
                signal_1n=signal1./max(abs(signal1));
                signal_2n=signal2./max(abs(signal2));

                % 
                V1_t=offs1+am1.*signal_1n;
                V2_t=offs2+am2.*signal_2n;
                
                %
                
                
                % 
                

                for nt=1:length(t)

                [V1a,nV1_a]=min(abs(V1_t(nt)-V1I));
                [V2a,nV2_a]=min(abs(V2_t(nt)-V2I));

                Rs11_t(nt)=real(s11_CI(nV2_a,nV1_a));
                Rs21_t(nt)=real(s21_CI(nV2_a,nV1_a));
                Is11_t(nt)=imag(s11_CI(nV2_a,nV1_a));
                Is21_t(nt)=imag(s21_CI(nV2_a,nV1_a));

                V1IT(nt)=V1I(nV1_a);
                V2IT(nt)=V2I(nV2_a); 
                end


                if fit==1
                [pl,rs]=polyfit(t,Rs11_t,40);
                [Rs11_tfit,delta]=polyval(pl,t,rs);

                [pl,rs]=polyfit(t,Rs21_t,40);
                [Rs21_tfit,delta]=polyval(pl,t,rs);

                [pl,rs]=polyfit(t,Is11_t,40);
                [Is11_tfit,delta]=polyval(pl,t,rs);

                [pl,rs]=polyfit(t,Is21_t,40);
                [Is21_tfit,delta]=polyval(pl,t,rs);

                end


                t0=-0.5/Om:1e-3/Om:0.5/Om-1e-3/Om;
                
                if fit==1
                s11_t=Rs11_tfit(501:1500)+1i.*Is11_tfit(501:1500);
                s21_t=Rs21_tfit(501:1500)+1i.*Is21_tfit(501:1500);

                s11_t=abs(s11_t).*exp(1i.*angle(s11_t));
                s21_t=abs(s21_t).*exp(1i.*angle(s21_t));
                
                else
                s11_t=Rs11_t(501:1500)+1i.*Is11_t(501:1500);
                s21_t=Rs21_t(501:1500)+1i.*Is21_t(501:1500);

                s11_t=abs(s11_t).*exp(1i.*angle(s11_t));
                s21_t=abs(s21_t).*exp(1i.*angle(s21_t));
                end

                % calculation of the Fourier components of S parameters 
                for n=0:1:nmax

                fa=cos(2*pi.*Om.*n.*t0);
                fb=sin(2*pi.*Om.*n.*t0);
                dx=2*pi*Om.*(t0(2)-t0(1));


                a_rf(n+1)=1/pi.*sum(s11_t.*fa)*dx;
                b_rf(n+1)=1/pi.*sum(s11_t.*fb)*dx;
                a_tr(n+1)=1/pi.*sum(s21_t.*fa)*dx;
                b_tr(n+1)=1/pi.*sum(s21_t.*fb)*dx;


                end


                s11_p=0.5*(a_rf-1i.*b_rf);
                s11_n=0.5*(a_rf+1i.*b_rf);
                s21_p=0.5*(a_tr-1i.*b_tr);
                s21_n=0.5*(a_tr+1i.*b_tr);



                As11=[fliplr(s11_p(2:end)),s11_n];
                As21=[fliplr(s21_p(2:end)),s21_n];
                
                
                As21_linear=As21;
                As11_linear=As11;
                 
                %figure(1) 
                %plot(spec);hold on;
                %plot(peakposition,As21,'o');
                
                %%%% objective function define the order of harmonic that requires maximum
                %%%% efficiency. -As11_linear(nmax+1) is to substract 0
                %%%% order if we only care single sideband
                OBJ11(n_geno)=abs(As21_linear(n_objhm))./abs(As21_linear(n_objhm+2));
                OBJ21(n_geno)=(abs(As21_linear(n_objhm))^2+1.*abs(As21_linear(n_objhm+2))^2)./(sum(abs(As21_linear).^2+abs(As11_linear).^2)); 
                
                OBJ(n_geno)=min(OBJ11(n_geno),1/OBJ11(n_geno))*OBJ21(n_geno);
                % Record the waveform parameters
                A1(n_geno,:)=a1;B1(n_geno,:)=b1;PH1(n_geno,:)=ph1;AM1(n_geno,:)=am1;OF1(n_geno,:)=offs1;
                A2(n_geno,:)=a2;B2(n_geno,:)=b2;PH2(n_geno,:)=ph2;AM2(n_geno,:)=am2;OF2(n_geno,:)=offs2;

                % Record the spectrum
                As21_Gn(n_geno,:)=As21;
                As11_Gn(n_geno,:)=As11;

                end
         
                
        % sort the objective function        
        [Sobj,Cn]=sort(OBJ);

        %%% Sort the parameter according the objective function achieved
        SortA1(:,:)=A1(Cn,:).'; 
        SortB1(:,:)=B1(Cn,:).';
        SortPH1(:,:)=PH1(Cn,:).'; 
        SortAM1(:,:)=AM1(Cn,:).'; 
        SortOF1(:,:)=OF1(Cn,:).'; 
        
        SortA2(:,:)=A2(Cn,:).'; 
        SortB2(:,:)=B2(Cn,:).'; 
        SortPH2(:,:)=PH2(Cn,:).'; 
        SortAM2(:,:)=AM2(Cn,:).';
        SortOF2(:,:)=OF2(Cn,:).'; 
        
        % record the best spectrum for each generation 
        BestOBJ(n_gr)=Sobj(end);
        Best_As21(n_gr,:)=As21_Gn(Cn(end),:);
        Best_As11(n_gr,:)=As11_Gn(Cn(end),:);

        % record the best parameter for each generation
        Best_A1(:,n_gr)=SortA1(:,end);
        Best_B1(:,n_gr)=SortB1(:,end);
        Best_PH1(:,n_gr)=SortPH1(:,end);
        Best_AM1(:,n_gr)=SortAM1(:,end);
        Best_OF1(:,n_gr)=SortOF1(:,end);
        
        Best_A2(:,n_gr)=SortA2(:,end);
        Best_B2(:,n_gr)=SortB2(:,end);
        Best_PH2(:,n_gr)=SortPH2(:,end);
        Best_AM2(:,n_gr)=SortAM2(:,end);
        Best_OF2(:,n_gr)=SortOF2(:,end);
        

        %%% Choose the best four parameters for the current generation
        BestA1(1:4,:)=SortA1(:,end-3:end).'; 
        BestB1(1:4,:)=SortB1(:,end-3:end).'; 
        BestPH1(1:4,:)=SortPH1(:,end-3:end).'; 
        BestAM1(1:4,:)=SortAM1(:,end-3:end).';
        BestOF1(1:4,:)=SortOF1(:,end-3:end).';
        
        BestA2(1:4,:)=SortA2(:,end-3:end).'; 
        BestB1(1:4,:)=SortB2(:,end-3:end).'; 
        BestPH2(1:4,:)=SortPH2(:,end-3:end).'; 
        BestAM2(1:4,:)=SortAM2(:,end-3:end).'; 
        BestOF2(1:4,:)=SortOF2(:,end-3:end).';
        
        %%% Generate new generation via hybridization
        
        % waveform 1
        NA1(1,:)=(BestA1(1,:)+BestA1(2,:))/2;NA1(2,:)=(BestA1(2,:)+BestA1(3,:))/2;
        NA1(3,:)=(BestA1(3,:)+BestA1(4,:))/2;NA1(4,:)=(BestA1(4,:)+BestA1(1,:))/2;
        
        NB1(1,:)=(BestB1(1,:)+BestB1(2,:))/2;NB1(2,:)=(BestB1(2,:)+BestB1(3,:))/2;
        NB1(3,:)=(BestB1(3,:)+BestB1(4,:))/2;NB1(4,:)=(BestB1(4,:)+BestB1(1,:))/2;

        NPH1(1,:)=(BestPH1(1,:)+BestPH1(2,:))/2;NPH1(2,:)=(BestPH1(2,:)+BestPH1(3,:))/2;
        NPH1(3,:)=(BestPH1(3,:)+BestPH1(4,:))/2;NPH1(4,:)=(BestPH1(4,:)+BestPH1(1,:))/2;

        NAM1(1,:)=(BestAM1(1,:)+BestAM1(2,:))/2;NAM1(2,:)=(BestAM1(2,:)+BestAM1(3,:))/2;
        NAM1(3,:)=(BestAM1(3,:)+BestAM1(4,:))/2;NAM1(4,:)=(BestAM1(4,:)+BestAM1(1,:))/2;
        
        NOF1(1,:)=(BestOF1(1,:)+BestOF1(2,:))/2;NOF1(2,:)=(BestOF1(2,:)+BestOF1(3,:))/2;
        NOF1(3,:)=(BestOF1(3,:)+BestOF1(4,:))/2;NOF1(4,:)=(BestOF1(4,:)+BestOF1(1,:))/2;
        
        NA1(5:end,:)=SortA1(:,5:end).';
        NB1(5:end,:)=SortB1(:,5:end).';
        NPH1(5:end,:)=SortPH1(:,5:end).';
        NAM1(5:end,:)=SortAM1(:,5:end).';
        NOF1(5:end,:)=SortOF1(:,5:end).';
        
                
        % waveform 2
        NA2(1,:)=(BestA2(1,:)+BestA2(2,:))/2;NA2(2,:)=(BestA2(2,:)+BestA2(3,:))/2;
        NA2(3,:)=(BestA2(3,:)+BestA2(4,:))/2;NA2(4,:)=(BestA2(4,:)+BestA2(1,:))/2;
        
        NB2(1,:)=(BestB2(1,:)+BestB2(2,:))/2;NB1(2,:)=(BestB2(2,:)+BestB2(3,:))/2;
        NB2(3,:)=(BestB2(3,:)+BestB2(4,:))/2;NB1(4,:)=(BestB2(4,:)+BestB2(1,:))/2;

        NPH2(1,:)=(BestPH2(1,:)+BestPH2(2,:))/2;NPH2(2,:)=(BestPH2(2,:)+BestPH2(3,:))/2;
        NPH2(3,:)=(BestPH2(3,:)+BestPH2(4,:))/2;NPH2(4,:)=(BestPH2(4,:)+BestPH2(1,:))/2;

        NAM2(1,:)=(BestAM2(1,:)+BestAM2(2,:))/2;NAM2(2,:)=(BestAM2(2,:)+BestAM2(3,:))/2;
        NAM2(3,:)=(BestAM2(3,:)+BestAM2(4,:))/2;NAM2(4,:)=(BestAM2(4,:)+BestAM2(1,:))/2;
        
        NA2(5:end,:)=SortA2(:,5:end).';
        NB2(5:end,:)=SortB2(:,5:end).';
        NPH2(5:end,:)=SortPH2(:,5:end).';
        NAM2(5:end,:)=SortAM2(:,5:end).';
        NOF2(5:end,:)=SortOF2(:,5:end).';
        
        
        if n_gr>4 
            
            % sort all the best geno achieved in each generation so far
            [S_bestobj,Cnbest]=sort(BestOBJ);
            
            Bestsofar_A1=Best_A1(:,Cnbest);
            Bestsofar_B1=Best_B1(:,Cnbest);
            Bestsofar_PH1=Best_PH1(:,Cnbest);
            Bestsofar_AM1=Best_AM1(:,Cnbest);
            Bestsofar_OF1=Best_OF1(:,Cnbest);
            
            Bestsofar_A2=Best_A2(:,Cnbest);
            Bestsofar_B2=Best_B2(:,Cnbest);
            Bestsofar_PH2=Best_PH2(:,Cnbest);
            Bestsofar_AM2=Best_AM2(:,Cnbest);
            Bestsofar_OF2=Best_OF2(:,Cnbest);
            
            % choose the best 4 as source of mutation for the new
            % generation, this can avoid degration if objective function
            % becomes worse as the system evolves
            
        NA1(5:8,:)=  Bestsofar_A1(:,end-3:end).';
        NB1(5:8,:)=  Bestsofar_B1(:,end-3:end).';
        NPH1(5:8,:)=  Bestsofar_PH1(:,end-3:end).';
        NAM1(5:8,:)=  Bestsofar_AM1(:,end-3:end).';
        NOF1(5:8,:)=  Bestsofar_OF1(:,end-3:end).';
        
        NA2(5:8,:)=  Bestsofar_A2(:,end-3:end).';
        NB2(5:8,:)=  Bestsofar_B2(:,end-3:end).';
        NPH2(5:8,:)=  Bestsofar_PH2(:,end-3:end).';
        NAM2(5:8,:)=  Bestsofar_AM2(:,end-3:end).';
        NOF2(5:8,:)=  Bestsofar_OF2(:,end-3:end).';
            
                              
        end
        
        
        if pn>1 && 9+pn-2<=N_Geno
            
            
        NA1(9:9+pn-2,:)=A1_best_allpass(1:pn-1,:);
        NB1(9:9+pn-2,:)=B1_best_allpass(1:pn-1,:);
        NPH1(9:9+pn-2,:)=PH1_best_allpass(1:pn-1,:);
        NAM1(9:9+pn-2,:)=AM1_best_allpass(1:pn-1,:);
        NOF1(9:9+pn-2,:)=OF1_best_allpass(1:pn-1,:);
        
        NA2(9:9+pn-2,:)=A2_best_allpass(1:pn-1,:);
        NB2(9:9+pn-2,:)=B2_best_allpass(1:pn-1,:);
        NPH2(9:9+pn-2,:)=PH2_best_allpass(1:pn-1,:);
        NAM2(9:9+pn-2,:)=AM2_best_allpass(1:pn-1,:);
        NOF2(9:9+pn-2,:)=OF2_best_allpass(1:pn-1,:); 
        
        elseif pn>1 && 9+pn-2>N_Geno
        
        NA1(9:end,:)=A1_best_allpass(pn+8-N_Geno:pn-1,:);
        NB1(9:end,:)=B1_best_allpass(pn+8-N_Geno:pn-1,:);
        NPH1(9:end,:)=PH1_best_allpass(pn+8-N_Geno:pn-1,:);
        NAM1(9:end,:)=AM1_best_allpass(pn+8-N_Geno:pn-1,:);
        NOF1(9:end,:)=OF1_best_allpass(pn+8-N_Geno:pn-1,:);
        
        NA2(9:end,:)=A2_best_allpass(pn+8-N_Geno:pn-1,:);
        NB2(9:end,:)=B2_best_allpass(pn+8-N_Geno:pn-1,:);
        NPH2(9:end,:)=PH2_best_allpass(pn+8-N_Geno:pn-1,:);
        NAM2(9:end,:)=AM2_best_allpass(pn+8-N_Geno:pn-1,:);
        NOF2(9:end,:)=OF2_best_allpass(pn+8-N_Geno:pn-1,:);     
                        
        end
        
        % draw the best waveform and spectrum
        
            for n=1:n_od 
                    sign1_best(n,:)=flag_sym.*Best_A1(n,n_gr).*sin(n.*(2*pi.*Om.*t+Best_PH1(n_gr)))+Best_B1(n,n_gr).*cos(n.*(2*pi.*Om.*t+Best_PH1(n_gr))); 
                    
                    if iden_waveform==1
                    
                    Best_A2(n,n_gr)=Best_A1(n,n_gr);
                    Best_B2(n,n_gr)=Best_B1(n,n_gr);
                    Best_PH2(n_gr)=Best_PH2(n_gr);
                    
                    end
                
                    
                    sign2_best(n,:)=flag_sym.*Best_A2(n,n_gr).*sin(n.*(2*pi.*Om.*t+Best_PH2(n_gr)))+Best_B2(n,n_gr).*cos(n.*(2*pi.*Om.*t+Best_PH2(n_gr)));
                    
            end
                
                % total signal
                signal1_best=sum(sign1_best);
                signal2_best=sum(sign2_best);
                % normalized signal
                signal_1n_best=signal1_best./max(abs(signal1_best));
                signal_2n_best=signal2_best./max(abs(signal2_best));
                
                V1_t=Best_OF1(n_gr)+Best_AM1(n_gr).*signal_1n;
                V2_t=Best_OF2(n_gr)+Best_AM2(n_gr).*signal_2n;
        
                figure(120)
                subplot(1,3,1)
                plot(t,V1_t,'r');hold on;
                plot(t,V2_t,'b');hold off;

                subplot(1,3,2)
                stem(abs(Best_As21(n_gr,:)).^2,'*')
                grid on
                
                subplot(1,3,3)
                stem(abs(Best_As11(n_gr,:)).^2,'*')
                grid on
                
                
                figure (121)
                plot(BestOBJ,'-o'); 
        
        end

    %
    [MM,bn]=max(BestOBJ);

    BestOBJ_allpass(pn,:)=MM;
    AS21_best_allpass(pn,:)=Best_As21(bn,:);
    AS11_best_allpass(pn,:)=Best_As11(bn,:);
    
    A1_best_allpass(pn,:)=Best_A1(:,bn).';
    B1_best_allpass(pn,:)=Best_B1(:,bn).';
    PH1_best_allpass(pn,:)=Best_PH1(:,bn).';
    AM1_best_allpass(pn,:)=Best_AM1(:,bn).';
    OF1_best_allpass(pn,:)=Best_OF1(:,bn).';
    
    A2_best_allpass(pn,:)=Best_A2(:,bn).';
    B2_best_allpass(pn,:)=Best_B2(:,bn).';
    PH2_best_allpass(pn,:)=Best_PH2(:,bn).';
    AM2_best_allpass(pn,:)=Best_AM2(:,bn).';
    OF2_best_allpass(pn,:)=Best_OF2(:,bn).';
end

    
   
 

%%
dphi=[0:1:0].*pi/180;

for n_dph=1:length(dphi)
    
    dPH=dphi(n_dph); 
    
  for pn=1:n_round
    
    
        for n=1:n_od 
            sign1_od(n,:)=flag_sym.*A1_best_allpass(pn,n).*sin(n.*(2*pi.*Om.*t+PH1_best_allpass(pn)))+B1_best_allpass(pn,n).*cos(n.*(2*pi.*Om.*t+PH1_best_allpass(pn))); 
            
            if iden_waveform==1
               A2_best_allpass(pn,n)=A1_best_allpass(pn,n); B2_best_allpass(pn,n)=B1_best_allpass(pn,n);
                
            end
            sign2_od(n,:)=flag_sym.*A2_best_allpass(pn,n).*sin(n.*(2*pi.*Om.*t+PH2_best_allpass(pn)+dPH))+B2_best_allpass(pn,n).*cos(n.*(2*pi.*Om.*t+PH2_best_allpass(pn)+dPH));
        end

  % total signal
                signal1_best_allpass=sum(sign1_od);
                signal2_best_allpass=sum(sign2_od);
                % normalized signal
                signal_1n_best_allpass=signal1_best_allpass./max(abs(signal1_best_allpass));
                signal_2n_best_allpass=signal2_best_allpass./max(abs(signal2_best_allpass));

                % 
                V1_best_allpass(pn,:)=OF1_best_allpass(pn,:)+AM1_best_allpass(pn,:).*signal_1n_best_allpass;
                V2_best_allpass(pn,:)=OF2_best_allpass(pn,:)+AM2_best_allpass(pn,:).*signal_2n_best_allpass;
                
                %  (iden_waveform.*signal_1n_best_allpass+(1-iden_waveform).*signal_2n_best_allpass)
                     
        for nt=1:length(t)

                [V1a,nV1_a]=min(abs(V1_best_allpass(pn,nt)-V1I));
                [V2a,nV2_a]=min(abs(V2_best_allpass(pn,nt)-V2I));

                Rs11_best_allpass(pn,nt)=real(s11_CI(nV2_a,nV1_a));
                Rs21_best_allpass(pn,nt)=real(s21_CI(nV2_a,nV1_a));
                Is11_best_allpass(pn,nt)=imag(s11_CI(nV2_a,nV1_a));
                Is21_best_allpass(pn,nt)=imag(s21_CI(nV2_a,nV1_a));


        end
        
        
                if fit==1
                [pl,rs]=polyfit(t,Rs11_best_allpass,40);
                [Rs11_tfit,delta]=polyval(pl,t,rs);

                [pl,rs]=polyfit(t,Rs21_best_allpass,40);
                [Rs21_tfit,delta]=polyval(pl,t,rs);

                [pl,rs]=polyfit(t,Is11_best_allpass,40);
                [Is11_tfit,delta]=polyval(pl,t,rs);

                [pl,rs]=polyfit(t,Is21_best_allpass,40);
                [Is21_tfit,delta]=polyval(pl,t,rs);

                s11_best_allpass_fit(pn,:)=Rs11_tfit(:,501:1500)+1i.*Is11_tfit(:,501:1500);
                s21_best_allpass_fit(pn,:)=Rs21_tfit(:,501:1500)+1i.*Is21_tfit(:,501:1500);

                end 
                    
                s11_best_allpass_t(pn,:)=Rs11_best_allpass(pn,501:1500)+1i.*Is11_best_allpass(pn,501:1500);
                s21_best_allpass_t(pn,:)=Rs21_best_allpass(pn,501:1500)+1i.*Is21_best_allpass(pn,501:1500);


% calculation of the Fourier components of S parameters 
                for n=0:1:nmax

                fa=cos(2*pi.*Om.*n.*t0);
                fb=sin(2*pi.*Om.*n.*t0);
                dx=2*pi*Om.*(t0(2)-t0(1));


                a_rf(n+1)=1/pi.*sum(s11_best_allpass_t(pn,:).*fa)*dx;
                b_rf(n+1)=1/pi.*sum(s11_best_allpass_t(pn,:).*fb)*dx;
                a_tr(n+1)=1/pi.*sum(s21_best_allpass_t(pn,:).*fa)*dx;
                b_tr(n+1)=1/pi.*sum(s21_best_allpass_t(pn,:).*fb)*dx;


                end


                s11_p=0.5*(a_rf-1i.*b_rf);
                s11_n=0.5*(a_rf+1i.*b_rf);
                s21_p=0.5*(a_tr-1i.*b_tr);
                s21_n=0.5*(a_tr+1i.*b_tr);



                As11_best_allpass_dphi0(pn,n_dph,:)=[fliplr(s11_p(2:end)),s11_n];
                As21_best_allpass_dphi0(pn,n_dph,:)=[fliplr(s21_p(2:end)),s21_n];
                
                
                As11_best_allpass_dphi(pn,n_dph,:)=abs(As11_best_allpass_dphi0(pn,n_dph,:)).^2./sum(abs(As11_best_allpass_dphi0(pn,n_dph,:)).^2+abs(As21_best_allpass_dphi0(pn,n_dph,:)).^2);
                As21_best_allpass_dphi(pn,n_dph,:)=abs(As21_best_allpass_dphi0(pn,n_dph,:)).^2./sum(abs(As11_best_allpass_dphi0(pn,n_dph,:)).^2+abs(As21_best_allpass_dphi0(pn,n_dph,:)).^2);
                
               
  end
                
end
                
%%

N=-nmax:1:nmax;
[MM,nbest]=max(BestOBJ_allpass);
 
n_dph=1;

As11_dphi(:,:)=As11_best_allpass_dphi(nbest,:,:);
As21_dphi(:,:)=As21_best_allpass_dphi(nbest,:,:);

%figure(190)

%plot(As11_dphi(:,nmax),'-*');hold on;
%plot(As11_dphi(:,nmax+2),'-*');hold on;
%plot(As21_dphi(:,nmax),'-o');hold on;
%plot(As21_dphi(:,nmax+2),'-o');hold off;

%
%As11_dphi_n1=As11_dphi(:,nmax);
%As11_dphi_p1=As11_dphi(:,nmax+2);
%As21_dphi_n1=As21_dphi(:,nmax);
%As21_dphi_p1=As21_dphi(:,nmax+2);

save As11_dph_sinle_n1.txt -ascii As11_dphi_n1
save As11_dph_sinle_p1.txt -ascii As11_dphi_p1
save As21_dph_sinle_n1.txt -ascii As21_dphi_n1
save As21_dph_sinle_p1.txt -ascii As21_dphi_p1

%%


figure(102)
subplot(1,2,1)
semilogy(N,abs(AS11_best_allpass(nbest,:)).^2,'*'); hold on;
semilogy(N,abs(AS21_best_allpass(nbest,:)).^2,'o'); hold off;
xlabel('order N');ylabel('scattering amplitude')
%legend('0','1','2','3','4','5','6')

subplot(1,2,2)
stem(N,-abs(AS11_best_allpass(nbest,:)).^2,'*'); hold on;
stem(N,abs(AS21_best_allpass(nbest,:)).^2,'o'); hold off;
xlabel('order N');ylabel('scattering amplitude')
%legend('0','1','2','3','4','5','6')
 

figure(104)
subplot(1,2,1)
plot(t,V1_best_allpass(nbest,:));hold on;
plot(t,V2_best_allpass(nbest,:),'-.');hold off;
ylabel('Capacitance');

subplot(1,2,2)
plot(t,V1_best_allpass(nbest,:));hold on;
plot(t,V2_best_allpass(nbest,:),'-.');hold off;

ylabel('V');
legend('V1','V2');


figure(105)
plot(1:N_GR,BestOBJ,'-*'); hold off;
xlabel('number of generation');ylabel('Energy ratio of \pm 1 orders')

figure(222)
plot(180.*(PH2_best_allpass-PH1_best_allpass)/pi,'*-')

figure(205)
subplot(2,2,1)
sf=contour(V1I,V2I,abs(s11_CI),15);
%set(sf, 'EdgeColor', 'none');
xlabel('C1(pF)');
ylabel('C2(pF)');
axis([min(V1I),max(V1I),min(V2I),max(V2I)])
zlabel('s11');
view([0,90])
tt=horzcat('|S11|, ','Freq=',num2str(ff),' GHz'); 
colorbar;hold on;
plot(V1_best_allpass(nbest,:),V2_best_allpass(nbest,:),'r');hold off
title(tt)

subplot(2,2,2)
sf=contour(V1I,V2I,abs(s21_CI),15);
%set(sf, 'EdgeColor', 'none');
xlabel('C1(pF)');
ylabel('C2(pF)');
axis([min(V1I),max(V1I),min(V2I),max(V2I)])
zlabel('s21');
view([0,90])
tt=horzcat('|S21|, ','Freq=',num2str(ff),' GHz'); 
colorbar;hold on 
title(tt)
plot(V1_best_allpass(nbest,:),V2_best_allpass(nbest,:),'r');hold off
 
subplot(2,2,3)
plot(t(1:1000),abs(s11_best_allpass_t(nbest,:)),'r');hold on;
plot(t(1:1000),abs(s21_best_allpass_t(nbest,:)),'b');hold off;
legend('|s11|','|s21|')

subplot(2,2,4)
plot(t(1:1000),angle(s11_best_allpass_t(nbest,:)),'r');hold on;
plot(t(1:1000),angle(s21_best_allpass_t(nbest,:)),'b');hold off;
legend('arg(s11)','arg(s21)')
%%
 


%% 

AS11_best=(abs(AS11_best_allpass(nbest,:)).^2./sum(abs(AS11_best_allpass(nbest,:)).^2+abs(AS21_best_allpass(nbest,:)).^2)).';

AS21_best=(abs(AS21_best_allpass(nbest,:)).^2./sum(abs(AS11_best_allpass(nbest,:)).^2+abs(AS21_best_allpass(nbest,:)).^2)).';

V1_waveform=V1_best_allpass(nbest,:).';
V2_waveform=V2_best_allpass(nbest,:).';




s21_best=abs(s21_best_allpass_t(nbest,:)).';
s11_best=abs(s11_best_allpass_t(nbest,:)).';
s21ph_best=180.*angle(s21_best_allpass_t(nbest,:)).'/pi;
s11ph_best=180.*angle(s11_best_allpass_t(nbest,:)).'/pi;

%

figure(12)
subplot(1,2,1)
semilogy(N,AS11_best,'o');hold on;
semilogy(N,AS21_best,'*');hold off;

subplot(1,2,2)
plot(N,AS11_best,'o');hold on;
plot(N,AS21_best,'*');hold off;
%%
save AS11_best_double_bi.txt -ascii AS11_best
save AS21_best_double_bi.txt -ascii AS21_best

save V1_best_double_bi.txt -ascii V1_waveform
save V2_best_double_bi.txt -ascii V2_waveform

save s21ph_best_double_bi.txt -ascii s21ph_best
save s11ph_best_double_bi.txt -ascii s11ph_best
save s21_best_double_bi.txt -ascii s21_best
save s11_best_double_bi.txt -ascii s11_best


%%
S11ph_V1_V2=180.*angle(s11_CI)/pi;
S21ph_V1_V2=180.*angle(s21_CI)/pi;
S11_V1_V2=abs(s11_CI);
S21_V1_V2=abs(s21_CI);

save S21_V1_V2.txt -ascii S21_V1_V2
save S11_V1_V2.txt -ascii S11_V1_V2
save S21ph_V1_V2.txt -ascii S21ph_V1_V2
save S11ph_V1_V2.txt -ascii S11ph_V1_V2