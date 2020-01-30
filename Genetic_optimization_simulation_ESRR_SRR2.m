clear;
clc
       ep0 = 8.854418e-15; %%%%%%%% unit: C/(V*mm)
        u0 = 4*pi*1e-10; %%%%%%%% unit: V*s/(A*mm)
        c=(ep0*u0)^(-0.5);  %%%%%% unit: mm/s
        
        

s21fin1=fopen('s21_ESRR+SRR_lossless.txt');
s21phfin1=fopen('s21phase_ESRR+SRR_lossless.txt');
s11fin1=fopen('s11_ESRR+SRR_lossless.txt');
s11phfin1=fopen('s11phase_ESRR+SRR_lossless.txt');


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
npar=2025;npar1=45;

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




C1=0.8:0.05:3;
C2=0.8:0.05:3;
C1I=0.8:0.001:3;
C2I=0.8:0.001:3;

S21_cc=s21_cc.*exp(-1i.*s21ph_cc.*pi/180);
S11_cc=s11_cc.*exp(-1i.*s11ph_cc.*pi/180);
   
   
ff=3.8;
[dd,fi]=min(abs(F-ff));
    
   
    Rs21c(:,:)=real(S21_cc(fi,:,:));
    Is21c(:,:)=imag(S21_cc(fi,:,:));
    Rs11c(:,:)=real(S11_cc(fi,:,:));
    Is11c(:,:)=imag(S11_cc(fi,:,:));
    
    
 
    
    for j=1:length(C2I)
        
        for k=1:length(C1)
            Rs21c_I(k,:)=interp1(C2,Rs21c(k,:),C2I,'spline');
            Is21c_I(k,:)=interp1(C2,Is21c(k,:),C2I,'spline');
            Rs11c_I(k,:)=interp1(C2,Rs11c(k,:),C2I,'spline');
            Is11c_I(k,:)=interp1(C2,Is11c(k,:),C2I,'spline');
        end
        
        Rs21_CI(:,j)=interp1(C1,Rs21c_I(:,j),C1I,'spline');
        Is21_CI(:,j)=interp1(C1,Is21c_I(:,j),C1I,'spline');
        Rs11_CI(:,j)=interp1(C1,Rs11c_I(:,j),C1I,'spline');
        Is11_CI(:,j)=interp1(C1,Is11c_I(:,j),C1I,'spline');
    end

 %%   
 %%%%% remove the additional phase accumulated along the waveguide
 dz=0;
 s11_CI=(Rs11_CI+1i.*Is11_CI).*exp(1i.*ff.*1e9*2*pi/c*dz);
 s21_CI=(Rs21_CI+1i.*Is21_CI).*exp(1i.*ff.*1e9*2*pi/c*dz); %%% forward scattering needs substract incident wave
 %%s21_CI=1-s11_CI;
 

Om=0.5e6;    %%%% Hz, frequency of modulation
t=-1.0/Om:1e-3/Om:1.0/Om;
c1_0=1.296;    %%%% center of biased capaciance
c2_0=1.09;
dc1=0.063*5;   %%%% amplitude of capaciance modulation 
dc2=0.03*5;
phi1=0.*pi/180;  %%%% relative phase of capaciance modulation 
phi2=0.*pi/180;

c1t=c1_0+dc1.*cos(2.*pi.*Om.*t+phi1); 
c2t=c2_0+dc2.*cos(2.*pi.*Om.*t+phi2);

%%%%%%%%%%%%%%%%%% voltage dependent capacitance
%Vf=1.8;c0=2.67;
%V1_0=-11.5;
%V2_0=-11.5;
%deltaV1=1;deltaV2=1.5;

%V1=V1_0+deltaV1.*cos(2.*pi.*Om.*t+phi1);
%V2=V2_0+deltaV2.*cos(2.*pi.*Om.*t+phi2);


%c1t=c0.*(1-V1./Vf).^(-0.5);
%c2t=c0.*(1-V2./Vf).^(-0.5);

figure(200)
subplot(1,3,1)
sf=contour(C1I,C2I,abs(s11_CI),15);
%set(sf, 'EdgeColor', 'none');
xlabel('C1(pF)');
ylabel('C2(pF)');
axis([min(C1),max(C1),min(C2),max(C2)])
zlabel('s11');
view([0,90])
tt=horzcat('|S11|, ','Freq=',num2str(ff),' GHz'); 
colorbar;hold on;
plot(c1t,c2t,'r');hold off
title(tt)

subplot(1,3,2)
sf=contour(C1I,C2I,abs(s21_CI),15);
%set(sf, 'EdgeColor', 'none');
xlabel('C1(pF)');
ylabel('C2(pF)');
axis([min(C1),max(C1),min(C2),max(C2)])
zlabel('s21');
view([0,90])
tt=horzcat('|S21|, ','Freq=',num2str(ff),' GHz'); 
colorbar;hold on 
title(tt)
plot(c1t,c2t,'r');hold off

subplot(1,3,3)
plot(t,c1t);hold on;
plot(t,c2t,'-.');hold off;
xlabel('t(s)');
ylabel('C1,C2(pF)');
legend('C1','C2')
colorbar;hold on;

figure(201)
subplot(1,3,1)
sf=contour(C1I,C2I,angle(s11_CI),15);
%set(sf, 'EdgeColor', 'none');
xlabel('C1(pF)');
ylabel('C2(pF)');
axis([min(C1),max(C1),min(C2),max(C2)])
zlabel('s11');
view([0,90])
tt=horzcat('arg(S11), ','Freq=',num2str(ff),' GHz'); 
colorbar;hold on;
plot(c1t,c2t,'r');hold off
title(tt)

subplot(1,3,2)
sf=contour(C1I,C2I,angle(s21_CI),15);
%set(sf, 'EdgeColor', 'none');
xlabel('C1(pF)');
ylabel('C2(pF)');
axis([min(C1),max(C1),min(C2),max(C2)])
zlabel('s21');
view([0,90])
tt=horzcat('arg(S21), ','Freq=',num2str(ff),' GHz'); 
colorbar;hold on 
title(tt)
plot(c1t,c2t,'r');hold off

subplot(1,3,3)
plot(t,c1t);hold on;
plot(t,c2t,'-.');hold off;
xlabel('t(s)');
ylabel('C1,C2(pF)');
legend('C1','C2')
colorbar;hold on;

%%
AM=2;

Om=0.5e6;    %%%% Hz, frequency of modulation
t=-1.0/Om:1e-3/Om:1.0/Om;
c1_0=1.296;    %%%% center of biased capaciance
c2_0=1.09;

phi1=0.*pi/180;  %%%% relative phase of capaciance modulation 
phi2=0.*pi/180;


a1=1;a2=0.0;a3=0.0;

for na=1:length(AM)
    
    am=AM(na);
    
dc1=0.063*am;   %%%% amplitude of capaciance modulation 
dc2=0.03*am;

Omg=2.*pi.*Om;
c1t=c1_0+dc1.*(a1*cos(Omg.*t+phi1)+a2*cos(2*Omg.*t+2*phi1)+a3*cos(3*Omg.*t+3*phi1)); 
c2t=c2_0+dc2.*(a1*cos(Omg.*t+phi2)+a2*cos(2*Omg.*t+2*phi2)+a3*cos(3*Omg.*t+3*phi2));


for nt=1:length(t)
    
[C1a,nC1_a]=min(abs(c1t(nt)-C1I));
[C2a,nC2_a]=min(abs(c2t(nt)-C2I));

Rs11_t(nt)=real(s11_CI(nC2_a,nC1_a));
Rs21_t(nt)=real(s21_CI(nC2_a,nC1_a));
Is11_t(nt)=imag(s11_CI(nC2_a,nC1_a));
Is21_t(nt)=imag(s21_CI(nC2_a,nC1_a));

C1IT(nt)=C1I(nC1_a);
C2IT(nt)=C2I(nC1_a); 
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




s11f=Rs11_t(501:1500)+1i.*Is11_t(501:1500);
s21f=Rs21_t(501:1500)+1i.*Is21_t(501:1500);

s11f=abs(s11f).*exp(1i.*angle(s11f));
s21f=abs(s21f).*exp(1i.*angle(s21f));


%%% calculate the fourier spectrum

%%
t_ssb=exp(1i.*2*pi.*Om.*t0);% t_ssb is to test single side band
t_ssb=exp(1i.*(pi.*sin(2*pi*Om.*t0)));% t_ssb is to test single side band
nmax=8;

for n=0:1:nmax

fa=cos(2*pi.*Om.*n.*t0);
fb=sin(2*pi.*Om.*n.*t0);
dx=2*pi*Om.*(t0(2)-t0(1));

 
a_rf(n+1)=1/pi.*sum(s11f.*fa)*dx;
b_rf(n+1)=1/pi.*sum(s11f.*fb)*dx;
a_tr(n+1)=1/pi.*sum(s21f.*fa)*dx;
b_tr(n+1)=1/pi.*sum(s21f.*fb)*dx;

a_tr_ssb(n+1)=1/pi.*sum(t_ssb.*fa)*dx;
b_tr_ssb(n+1)=1/pi.*sum(t_ssb.*fb)*dx;

end

s11_p=0.5*(a_rf+1i.*b_rf);
s11_n=0.5*(a_rf-1i.*b_rf);
s21_p=0.5*(a_tr+1i.*b_tr);
s21_n=0.5*(a_tr-1i.*b_tr);

s21_p_ssb=0.5*(a_tr_ssb+1i.*b_tr_ssb);
s21_n_ssb=0.5*(a_tr_ssb-1i.*b_tr_ssb);




As11=[fliplr(s11_p(2:end)),s11_n];
As21=[fliplr(s21_p(2:end)),s21_n];

As21_ssb=[fliplr(s21_p_ssb(2:end)),s21_n_ssb];

%AS11_AM(na,:)=As11;
%AS21_AM(na,:)=As21;
end
%%

N=-nmax:1:nmax;

figure(101)
stem(-nmax:1:nmax,abs(As21_ssb));hold on;
%stem(-nmax:1:nmax,abs(As21_ssb2));hold off;
%plot(AM,abs(AS21_AM(:,7:end)).^2);hold off;

%% optimization


warning on;
optimiz=1;
fit=0;

am=5.5;
dc1=0.063*am;   %%%% amplitude of capaciance modulation 
dc2=0.03*am;

phi1=0.*pi/180;  %%%% relative phase of capaciance modulation 
phi2=0.*pi/180;


%%% degree of mutation in amplitude and phases
mutam=0.5; 
mutph=5.*pi/180; 

n_od=3;  %%% number of orders in modulation
N_GR=40; %%% number of generation
N_Geno=40; %%% number of genos
nmax=20;     %%%% number of orders of harmonics for HB analysis
n_round=5;  %%% number of rounds of optimization

a1_0=[random('unif',-mutam,mutam,1,n_od-1),1,random('unif',-mutam,mutam,1,n_od-1)];  %%%% initial amplitude for different orders
a2_0=[random('unif',-mutam,mutam,1,n_od-1),1,random('unif',-mutam,mutam,1,n_od-1)];  %%%% initial amplitude for different orders

ph1_0=[random('unif',-mutph,mutph,1,n_od-1),phi1,random('unif',-mutph,mutph,1,n_od-1)];  %%%% initial amplitude for different orders
ph2_0=[random('unif',-mutph,mutph,1,n_od-1),phi2,random('unif',-mutph,mutph,1,n_od-1)];  %%%% initial amplitude for different orders

BestOBJ_allpass=zeros(n_round,1);
AS21_best_allpass=zeros(n_round,nmax*2+1); 
AS11_best_allpass=zeros(n_round,nmax*2+1); 

A_best_allpass=zeros(n_round,2*n_od-1); 

%NA1=zeros(1,N_Geno);NA2=zeros(1,N_Geno);NA3=zeros(1,N_Geno);NA4=zeros(1,N_Geno);NA5=zeros(1,N_Geno);
%SortA1=zeros(1,N_Geno);SortA2=zeros(1,N_Geno);SortA3=zeros(1,N_Geno);SortA4=zeros(1,N_Geno);SortA5=zeros(1,N_Geno);

for pn=1:n_round
    
    %%%% NA: new amplitude for different orders of modulation for different
    %%%% genos, it is generated by hybridiing the best 4 and sorting the
    %%%% other genos based on performance
    
NA=zeros(N_Geno,2*n_od-1);
NPH=zeros(N_Geno,2*n_od-1);

%%%% the matrix of sorted different orders of amplitude of modulation for different genos
%%%% according to the objective function
SortA=zeros(2*n_od-1,N_Geno);

%%% the objective function 
BestOBJ=zeros(1,N_GR);

%%% the S parameters of different orders of harmonics for different genos
As21_Gn=zeros(N_Geno,nmax*2+1);
As11_Gn=zeros(N_Geno,nmax*2+1);
%%% S parameters for the best geno in each generation
Best_As21=zeros(N_GR,nmax*2+1);
Best_As11=zeros(N_GR,nmax*2+1);
%%% amplitude for the best geno in each generation
Best_CA=zeros(2*n_od-1,N_GR);
Best_PH=zeros(2*n_od-1,N_GR);
%%% amplitude of modulation harmonics
A=zeros(N_GR,2*n_od-1);
%%% best four genos
BestA=zeros(4,2*n_od-1);
BestPH=zeros(4,2*n_od-1);
%%%% initial amplitude for different orders
a1_0=[random('unif',-mutam,mutam,1,n_od-1),1,random('unif',-mutam,mutam,1,n_od-1)];  %%%% initial amplitude for different orders
a2_0=[random('unif',-mutam,mutam,1,n_od-1),1,random('unif',-mutam,mutam,1,n_od-1)];  %%%% initial amplitude for different orders

ph1_0=[random('unif',-mutph,mutph,1,n_od-1),phi1,random('unif',-mutph,mutph,1,n_od-1)];  %%%% initial amplitude for different orders
ph2_0=[random('unif',-mutph,mutph,1,n_od-1),phi2,random('unif',-mutph,mutph,1,n_od-1)];  %%%% initial amplitude for different orders
 
%%% normalization 
%a_0=a_0./sqrt(sum(abs(a_0).^2)); 



if optimiz==1

    
    %%%%% Generation
for n_gr=1:1:N_GR

     %%%%% Geno
for n_geno=1:1:N_Geno
    
   
      %%% for the second generation use new genos formed based on the last
      %%% generation
    if n_gr>1  
     
        a_0=NA(n_geno,:);        
        %a_0=a_0./sqrt(sum(abs(a_0).^2));
        
    end
    
  Omg=2.*pi.*Om;  
    
  
  %%% for geno number larger than 5, introduce mutation
      if n_geno>4
        ca=a_0.*(1+random('unif',-mut,mut,1,2*n_od-1))+1.0.*random('unif',-mut,mut,1,2*n_od-1);
      else
        ca=a_0;
      end
      
      %ca=ca./sqrt(sum(abs(ca).^2));

      
      %%%% define the spectral component of modulation
      
for n=1:n_od 
c1t_od(n,:)=(-1)^(n)*ca(n).*sin(n.*(Omg.*t+phi1));  %%% (-1)^n is used to construct sawtooth fuction to achieve singal sideband
c2t_od(n,:)=(-1)^(n)*ca(n).*sin(n.*(Omg.*t+phi2));
end

c1t=c1_0+dc1.*sum(c1t_od(:,:));
c2t=c2_0+dc2.*sum(c2t_od(:,:));

for nt=1:length(t)
    
[C1a,nC1_a]=min(abs(c1t(nt)-C1I));
[C2a,nC2_a]=min(abs(c2t(nt)-C2I));

Rs11_t(nt)=real(s11_CI(nC2_a,nC1_a));
Rs21_t(nt)=real(s21_CI(nC2_a,nC1_a));
Is11_t(nt)=imag(s11_CI(nC2_a,nC1_a));
Is21_t(nt)=imag(s21_CI(nC2_a,nC1_a));

C1IT(nt)=C1I(nC1_a);
C2IT(nt)=C2I(nC1_a); 
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





s11f=Rs11_t(501:1500)+1i.*Is11_t(501:1500);
s21f=Rs21_t(501:1500)+1i.*Is21_t(501:1500);

s11f=abs(s11f).*exp(1i.*angle(s11f));
s21f=abs(s21f).*exp(1i.*angle(s21f));



for n=0:1:nmax

fa=cos(2*pi.*Om.*n.*t0);
fb=sin(2*pi.*Om.*n.*t0);
dx=2*pi*Om.*(t0(2)-t0(1));

 
a_rf(n+1)=1/pi.*sum(s11f.*fa)*dx;
b_rf(n+1)=1/pi.*sum(s11f.*fb)*dx;
a_tr(n+1)=1/pi.*sum(s21f.*fa)*dx;
b_tr(n+1)=1/pi.*sum(s21f.*fb)*dx;


end



s11_p=0.5*(a_rf+1i.*b_rf);
s11_n=0.5*(a_rf-1i.*b_rf);
s21_p=0.5*(a_tr+1i.*b_tr);
s21_n=0.5*(a_tr-1i.*b_tr);

N=-nmax:1:nmax;

As11=[fliplr(s11_p(2:end)),s11_n];
As21=[fliplr(s21_p(2:end)),s21_n];


%%%% objective function define the order of harmonic that requires maximum
%%%% efficiency. 
OBJ(n_geno)=2.*abs(As21(nmax))^2;%/sum(abs(As21(1:end)).^2+abs(As11(1:end)).^2);
if phi2==180.*pi/180;
OBJ(n_geno)=2.*abs(As11(nmax))^2;%/sum(abs(As21(1:end)).^2+abs(As11(1:end)).^2);
end
    
A(n_geno,:)=ca;


As21_Gn(n_geno,:)=As21;
As11_Gn(n_geno,:)=As11;

end

[Sobj,Cn]=sort(OBJ);


%%% Sort the parameter according the objective function achieved
SortA(:,:)=A(Cn,:).'; 

BestOBJ(n_gr)=Sobj(end);
Best_As21(n_gr,:)=As21_Gn(Cn(end),:);
Best_As11(n_gr,:)=As11_Gn(Cn(end),:);

Best_CA(:,n_gr)=SortA(:,end);


%%% Choose the best four parameters
BestA(1:4,:)=SortA(:,end-3:end).'; 


%%% Generate new generation
NA(1,:)=(BestA(1,:)+BestA(2,:))/2;NA(2,:)=(BestA(2,:)+BestA(3,:))/2;
NA(3,:)=(BestA(3,:)+BestA(4,:))/2;NA(4,:)=(BestA(4,:)+BestA(1,:))/2;
 
NA(5:end,:)=SortA(:,5:end).';


end


[MM,bn]=max(BestOBJ);

BestOBJ_allpass(pn,:)=MM;
AS21_best_allpass(pn,:)=Best_As21(bn,:);
AS11_best_allpass(pn,:)=Best_As11(bn,:);
A_best_allpass(pn,:)=Best_CA(:,bn).';

end

end

%%
[MMA,bn]=max(BestOBJ_allpass);
AS21_best=AS21_best_allpass(bn,:);
AS11_best=AS11_best_allpass(bn,:);
A_best=A_best_allpass(bn,:);

ca=A_best;

for n=1:n_od 
c1t_od(n,:)=(-1)^n.*ca(n).*cos(n.*(Omg.*t+phi1));
c2t_od(n,:)=(-1)^n.*ca(n).*cos(n.*(Omg.*t+phi2));
end


c1t_best=c1_0+dc1.*sum(c1t_od(:,:));
c2t_best=c2_0+dc2.*sum(c2t_od(:,:));

figure(102)
semilogy(N,abs(AS11_best),'*'); hold on;
semilogy(N,abs(AS21_best),'o'); hold off;
xlabel('order N');ylabel('scattering amplitude')
legend('0','1','2','3','4','5','6')

figure(103)
subplot(1,2,1)
plot(1:N_GR,abs(Best_As21(:,7:end)).^2,'-*'); hold off;
xlabel('number of generation');ylabel('|S21|^2')

subplot(1,2,2)
plot(1:N_GR,BestOBJ,'-*'); hold off;
xlabel('number of generation');ylabel('Energy ratio of \pm 1 orders')

figure(104)
plot(t,c1t_best);hold on;
plot(t,c2t_best,'-.');hold off;
%%
figure(105)
plot(1:N_GR,BestOBJ,'-*'); hold off;
xlabel('number of generation');ylabel('Energy ratio of \pm 1 orders')



%%
figure(111)
subplot(1,2,1)
plot(t,c1t);hold on;
plot(t,c2t,'-.');hold off;
xlabel('t(s)');
ylabel('C1,C2(pF)');
legend('C1','C2')

subplot(1,2,2)
plot(c1t,c2t);hold off;
axis([min(C1),max(C1),min(C2),max(C2)])
xlabel('C1(pF)');
ylabel('C2(pF)');
 
figure(30)
subplot(1,3,1)
plot(Om.*t.*2*pi,Rs11_t);hold on;
%plot(Om.*t.*2*pi,Rs11_tfit);hold on;
plot(Om.*t.*2*pi,Rs21_t);hold on;
%plot(Om.*t.*2*pi,Rs21_tfit);hold on;
plot(Om.*t.*2*pi,Is11_t);hold on;
%plot(Om.*t.*2*pi,Is11_tfit);hold on;
plot(Om.*t.*2*pi,Is21_t);hold on;
%plot(Om.*t.*2*pi,Is21_tfit);hold off;

legend('Re(s11)_fit','Re(s21)_fit','Im(s11)_fit','Im(s21)_fit')

subplot(1,3,2)
%plot(Om.*t.*2*pi,Rs11_t);hold on;
plot(Om.*t.*2,abs(Rs11_t+1i.*Is11_t));hold on;
%plot(Om.*t.*2*pi,Rs21_t);hold on;
plot(Om.*t.*2,abs(Rs21_t+1i.*Is21_t));hold off;
%%%Fourier transformation

subplot(1,3,3)
%plot(Om.*t.*2*pi,Rs11_t);hold on;
plot(Om.*t.*2,180/pi.*angle(Rs11_t+1i.*Is11_t));hold on;
%plot(Om.*t.*2*pi,Rs21_t);hold on;
plot(Om.*t.*2,180/pi.*angle(Rs21_t+1i.*Is21_t));hold off;
%%%Fourier transformation

figure(125)

subplot(1,3,1)
plot(t0,abs(s11f));hold on;
plot(t0,abs(s21f));hold off;
legend('|s11|','|s21|');
xlabel('t(s)');ylabel('scattering amplitude')


subplot(1,3,2)
plot(N,abs(As11).^2,'*'); hold on;
plot(N,abs(As21).^2,'o'); hold off;
xlabel('order N');ylabel('scattering amplitude')

legend('|s11|','|s21|');

subplot(1,3,3)
stem(N,(angle(As11)).*180/pi,'*'); hold on;
stem(N,(angle(As21)).*180/pi,'o'); hold off;
legend('arg(s11)','arg(s21)');
xlabel('order N');ylabel('scattering phase')




