%%% This is a program that controls 8 channels of signals from function generators, and scan the nearfield signals by controlling the translational stage that moves a parallel plate waveguide 


clear;

%%
warning on;
% close all open instruments
objs = instrfind;
if length(objs)>1
fclose(objs)
else fprint('no object connected')
end

%%
%%%% Arbitrary Function Generator connection and settings
    % assign visa of AFG
    vFG = visa('ni','GPIB0::11::INSTR');
    vFG2 = visa('ni','GPIB0::13::INSTR');
    vFG_R1 = visa( 'ni','USB0::0x1AB1::0x0642::DG1ZA193704182::0::INSTR'); %Create VISA object for RIGOL AFG 
    vFG_R2 = visa( 'ni','USB0::0x1AB1::0x0642::DG1ZA193704127::0::INSTR');

    % define buffersize for output 
    vFG.outputbuffersize = 10000;
    vFG2.outputbuffersize = 10000;
    vFG_R1.outputbuffersize = 10000;
    vFG_R2.outputbuffersize = 10000;
    % connect to AFG
    fopen(vFG);
    fopen(vFG2);
    fopen(vFG_R1);
    fopen(vFG_R2);
    
    %fclose(vFG);
    %fclose(vFG2);
    %fclose(vFG_R1);
    %fclose(vFG_R2);
    %%
    % Initialise AFG
    flag_master=0; % if flag_master=1, then it serves as the master of synchronisation
    Fun_ini_AFG_RIGOL(vFG_R1,flag_master);
    flag_master=1;
    Fun_ini_AFG_RIGOL(vFG_R2,flag_master);

%% Signal Analyzer connection and settings

    vSA = visa('ni','GPIB0::21::INSTR');
    vSA.inputbuffersize = 40000*40;
    fopen(vSA);

    cen_freq=4; % center frequency GHz
    freq_span=20; % measure frequency span kHz
    bandwidth_m=5; % bandwidth of measurement resolution Hz
    bandwidth_v=10; % bandwidth of video resolution Hz
    num_points=20001;% number of data points

    Fun_ini_SA(vSA,cen_freq,freq_span,bandwidth_m,bandwidth_v,num_points); % initialize signal analyzer

    % setup the datapoint positions for harmonics, this depends on frequency span, number
    % of datapoint and modulation frequency. Better to do a measurement first
    % and check the point position

    % first point of harmonic peak
    n_1=1;
    % seperation between harmonic peak
    dn=1000;
    % final point of harmonic peak
    n_end=num_points-n_1+1;
    % peak position in the measured data array
    peakposition=n_1:dn:n_end;
    % maximum order of harmonics
    nmax=(length(peakposition)-1)/2;
    % define which objective harmonic to be optimized
    n_objhm=nmax; % corresponds to -1 order

%%%% Create session to control switch USB6525

    % Add digital output channel
    sUSB=daq.createSession('ni');
    addDigitalChannel(sUSB,'Dev1','port0/line0','OutputOnly');
    addDigitalChannel(sUSB,'Dev1','port0/line1','OutputOnly');

%%%% Control of signal generator HP-8637B

    vSG = visa('ni','GPIB0::19::INSTR');
    vSG.inputbuffersize = 10000;
    fopen(vSG);
    %
    % power of CW signal in dBm
    power_dbm=10;
    % frequency of CW signal 
    cwfr=cen_freq;
    Fun_SG8673B(vSG,cen_freq,power_dbm);
    %fclose(vSG);

   %%  connect to translational stage and initialize

vTS = visa('ni','ASRL1::INSTR');
vTS.BaudRate=9600;
%vTS.Terminator='CR';
fopen(vTS);
vTS.timeout = 5;

%
Fun_ini_TS(vTS);
query = fscanf(vTS, '%s\r');

fprintf(vTS,'%s\r','0GH');  % GO HOME!
query = fscanf(vTS, '%s\r');


%%
%%%% setup time and period of modulation 
    %(nominal, doesn't really matter as long as the number of datapoints is 1000)
    % frequency
    Freq=1*1e3;  
    % time step
    dt=0.001/Freq;   
    % time
    t=0:dt:1/Freq-dt;  

    
%%%% A test of randomly changed waveforms and the corresponding spectra
flag_test=1;

specS21=zeros(num_points,5);
specS11=zeros(num_points,5);
    if flag_test==1
        tic
        for n=1:5

        % amplitude of coefficient
        a1=1; a2=1.0;
        % phase
        ph1=0; ph2=pi/3*random('unif',-1,1);
        % signal components
        signal_1=a1*sin(2*pi*Freq*t+ph1)+random('unif',-1,1)*sin(2*pi*2*Freq*t+2*ph1);
        signal_2=a2*sin(2*pi*Freq*t+ph2)+random('unif',-1,1)*sin(2*pi*2*Freq*t+2*ph2);


        % normalized signal to maximum amplitude 1
        signal_1n=1.*signal_1./max(abs(signal_1));
        signal_2n=1.*signal_2./max(abs(signal_2));

        % Vpp amplitude and offset of output voltage
        am1=2;am2=2;
        offset1=2.2;offset2=2.1;

        Fun_AFG2(vFG2,signal_1n,signal_2n,am1,am2,offset1,offset2);
        Fun_AFG(vFG,signal_1n,signal_2n,am1,am2,offset1,offset2);

        %fclose(vFG);

        %  for S21
        line0=1;line1=0;
        USB6525(sUSB,line0,line1);
        specS21(:,n)=Fun_SA(vSA,num_points);

        %  for S11
        line0=0;line1=1;
        USB6525(sUSB,line0,line1);
        specS11(:,n)=Fun_SA(vSA,num_points);



        end

        toc

        %%% Plot sideband signals
        xx=ones(num_points,n);
        yy=ones(num_points,n);

        for jj=1:n
        yy(:,jj)=jj;
        end

        for ii=1:num_points
        xx(ii,:)=ii;   
        end


        figure(1);
        subplot(1,2,1)
        plot3(xx,yy,specS21(:,1:n));

        subplot(1,2,2)
        plot3(xx,yy,specS11(:,1:n));
        hold off;

    end


%%
%%%%%%%%% OPTIMIZATION

%%%% initial input parameters of waveform

% amplitude of modulation (offset tuned manually)
am=[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5];
% relative phase
phi=[0,0,0,0,0,0,0,0].*pi/180;   
% offset
offs=[1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8];

%  flag_idenwf==1: v1 and v2 have identical waveform, except offset, amplitude and phase
flag_idenwf=0;

% flag_idensig=1: all channels for electric meta-atoms have identical signal, the same for magnetic meta-atoms 
flag_idensig=1;

%
flag_optimize=1;

%%% degree of mutation in amplitude and phases
mutam=0.1;  % mutation of amplitude of waveform
mutco=0.05;  % mutation of coefficient of harmonics of waveform 
mutph=1.*pi/180;  % mutation of phase of waveform
mutof=0.05;  % mutation of offset

%%% parameters of optimization
n_od=8;  %%% number of orders in modulation
N_GR=40; %%% number of generation
N_Geno=40; %%% number of genos
%nmax=8;     %%%% number of orders of harmonics for HB analysis
n_round=6;  %%% number of rounds of optimization
n_ch=8;  %%% number of output channel

%%% parameters of translational stage
THETA=(0:180:180).*pi/180;
R=150;
X=R.*cos(THETA+pi/4)+20;
Y=R.*sin(THETA+pi/4)+10;
N_Trans=length(X);

%
C_11=0; %%% correction term for S11. for 1d waveguide, C_11=18.2.

CWfreq=[4];  % frequencies of carrier wave 

%
for n_cw=1:length(CWfreq) 
    
    %%%% setup central frequency for signal analyzer and CW signal generator
    cwfreq=CWfreq(n_cw);
    FREQSA=['FREQ:CENT ',num2str(cwfreq),' GHz'];
    fwrite(vSA, FREQSA);  
    Fun_SG8673B(vSG,cwfreq,power_dbm);

    %%% Define the arrays for recording optimized parameters
    BestOBJ_allpass=zeros(n_round,1);

    AS21_best_allpass=zeros(n_round,N_Trans,nmax*2+1); 
    %AS11_best_allpass=zeros(n_round,nmax*2+1); 

    spec21_best_allpass=zeros(n_round,N_Trans,num_points); 
    %spec11_best_allpass=zeros(n_round,num_points); 

    A_best_allpass=zeros(n_ch,n_od,n_round); 
    B_best_allpass=zeros(n_ch,n_od,n_round); 
    PH_best_allpass=zeros(n_ch,1,n_round); 
    AM_best_allpass=zeros(n_ch,1,n_round); 
    OF_best_allpass=zeros(n_ch,1,n_round); 

    sign_od=zeros(n_ch,n_od,length(t));
    signal=zeros(n_ch,length(t));
    signal_n=zeros(n_ch,length(t)); 
  
    %%
    %%%%% Round
    for pn=1:n_round

        %%%% N*: new coefficients of modulation waveform for different
        %%%% genos, it is generated by hybridiing the best 6, which generate 15 new generation
        %%%% and sorting the other genos based on performance

        NA=zeros(n_ch,n_od,N_Geno);
        NB=zeros(n_ch,n_od,N_Geno); 
        NPH=zeros(n_ch,1,N_Geno);
        NAM=zeros(n_ch,1,N_Geno); 
        NOF=zeros(n_ch,1,N_Geno); 
    

        %%%% the matrix of sorted different orders of amplitude of modulation for different genos
        %%%% according to the objective function
        SortA=zeros(n_ch,n_od,N_Geno); 
        SortB=zeros(n_ch,n_od,N_Geno); 
        SortPH=zeros(n_ch,1,N_Geno);
        SortAM=zeros(n_ch,1,N_Geno);
        SortOF=zeros(n_ch,1,N_Geno);
        %%% the objective function 
        BestOBJ=zeros(1,N_GR);

        %%% the S parameters of different orders of harmonics for different genos
        As21_Gn=zeros(N_Geno,N_Trans,nmax*2+1);
        %As11_Gn=zeros(N_Geno,nmax*2+1);
        As21_linear_Gn=zeros(N_Geno,N_Trans,nmax*2+1);
        %%% spectra of different genos
        %spec11_Gn=zeros(N_Geno,num_points);
        spec21_Gn=zeros(N_Geno,N_Trans,num_points);

        %%% S parameters for the best geno in each generation
        Best_As21=zeros(N_GR,N_Trans,nmax*2+1);
        %Best_As11=zeros(N_GR,nmax*2+1);

        Best_spec21=zeros(N_GR,N_Trans,num_points);
        %Best_spec11=zeros(N_GR,num_points);
        %%% amplitude and phase for the best geno in each generation
        Best_A=zeros(n_ch,n_od,N_GR);
        Best_B=zeros(n_ch,n_od,N_GR);
        Best_PH=zeros(n_ch,1,N_GR);
        Best_AM=zeros(n_ch,1,N_GR);
        Best_OF=zeros(n_ch,1,N_GR);

        %%% Define amplitude of modulation harmonics
        A=zeros(n_ch,n_od,N_GR);B=zeros(n_ch,n_od,N_GR);
        PH=zeros(n_ch,1,N_GR);AM=zeros(n_ch,1,N_GR);OF=zeros(n_ch,1,N_GR);

        %%% the best 6 genos in each generation, will produce 15 new hybrid genos in the new generation 
        BestA=zeros(n_ch,n_od,6);
        BestB=zeros(n_ch,n_od,6);
        BestPH=zeros(n_ch,1,6);
        BestAM=zeros(n_ch,1,6);
        BestOF=zeros(n_ch,1,6);
    
        %%% define the coefficient of input waveform
        a_0=zeros(n_ch,n_od);
        b_0=zeros(n_ch,n_od);
        ph_0=zeros(n_ch,1);
        am_0=zeros(n_ch,1);
        offs_0=zeros(n_ch,1);
        
        
        
        for n=1:n_ch

        a_0(n,:)=[1,random('unif',-mutco,mutco,1,n_od-1)];  %%%% initial amplitude for different orders
        b_0(n,:)=[1,random('unif',-mutco,mutco,1,n_od-1)];  %%%% initial amplitude for different orders
        ph_0(n,:)=phi(n)+(n~=1).*random('unif',-mutph,mutph,1,1);  %%%% initial phase for different orders    
        am_0(n,:)=am(n).*(1+random('unif',-mutam,mutam,1,1));
        offs_0(n,:)=offs(n).*(1+random('unif',-mutof,mutof,1,1))+random('unif',-mutof,mutof,1,1);

        end


       %%%%% Generation 
       
        for n_gr=1:1:N_GR
                     

            for n_trans=1:1:N_Trans
                
                x=X(n_trans);y=Y(n_trans);
                [pos_x,pos_y]=Fun_scan_TS(vTS,x,y);
                
             %%%%% Geno
                for n_geno=1:1:N_Geno
                    
                    
                   
                  if n_trans==1     % if it is the first measurement position, use hybridized coefficient for waveform generation
                    
                          %%% for the second generation use new genos formed based on the last
                          %%% generation
                        if n_gr>1  
                            a_0(:,:)=NA(:,:,n_geno); 
                            b_0(:,:)=NB(:,:,n_geno); 
                            ph_0(:,:)=NPH(:,:,n_geno); 
                            am_0(:,:)=NAM(:,:,n_geno);
                            offs_0(:,:)=NOF(:,:,n_geno);
                        end


                      %%% for geno number larger than 15, introduce mutation
                          if n_geno>15

                              for n=1:n_ch
                                a(n,:)=a_0(n,:)+random('unif',-mutco,mutco,1,n_od);
                                b(n,:)=b_0(n,:)+random('unif',-mutco,mutco,1,n_od);
                                ph(n,:)=ph_0(n,:)+(n~=1).*random('unif',-mutph,mutph,1,1);
                                am(n,:)=am_0(n,:).*(1+random('unif',-mutam,mutam,1,1));
                                offs(n,:)=offs_0(n,:).*(1+random('unif',-mutof,mutof,1,1))+random('unif',-mutof,mutof,1,1);
                              end
                          % for geno number <6, use the hybrid geno without mutation
                          else
                            a=a_0; b=b_0;
                            ph=ph_0; am=am_0; offs=offs_0;

                          end



                        % define the spectral component of modulation

                            if flag_idenwf==1  % if flag_idenwf=1, then channel 2=channel 1, channel 4 = channel 3
                            a(2:2:end,:)=a(1:2:end,:); 
                            b(2:2:end,:)=b(1:2:end,:); 
                            end

                            if flag_idensig==1
                            a(3:2:end,:)=ones(3,1)*a(1,:); a(4:2:end,:)=ones(3,1)*a(2,:); 
                            b(3:2:end,:)=ones(3,1)*b(1,:); b(4:2:end,:)=ones(3,1)*b(2,:);
                            am(3:2:end)=am(1); am(4:2:end)=am(2);
                            ph(3:2:end)=ph(1);ph(4:2:end)=ph(2);
                            offs(3:2:end)=offs(1); offs(4:2:end)=offs(2);
                            end

                             % Record the waveform parameters
                            A(:,:,n_geno)=a;B(:,:,n_geno)=b;
                            PH(:,:,n_geno)=ph;AM(:,:,n_geno)=am;OF(:,:,n_geno)=offs;
                        
                  else  % otherwise, use the same coefficient for the same geno number
                      
                       a=A(:,:,n_geno);b=B(:,:,n_geno);
                       ph=PH(:,:,n_geno);am=AM(:,:,n_geno);offs=OF(:,:,n_geno);
                        
                  
                  end
                      
                  A_record(:,:,n_trans)=a;B_record(:,:,n_trans)=b;
                  PH_record(:,:,n_trans)=ph;AM_record(:,:,n_trans)=am;OF_record(:,:,n_trans)=offs;
                   
                  
                    %%% contruct signals
                    for nn=1:n_ch

                        for n=1:n_od 

                        sign_od(nn,n,:)=a(nn,n).*sin(n.*(2*pi.*Freq.*t+ph(nn,1)))+b(nn,n).*cos(n.*(2*pi.*Freq.*t+ph(nn,1)));  

                        end

                        signal(nn,:)=sum(sign_od(nn,:,:));
                        signal_n(nn,:)=signal(nn,:)./max(abs(signal(nn,:)));

                    end
                    
                    
                % 4 pairs
                % write signal to AFG                
                %Fun_AFG2(vFG2,signal_n(1,:),signal_n(2,:),am(1),am(2),offs(1),offs(2));
                Fun_AFG(vFG,signal_n(3,:),signal_n(4,:),am(3),am(4),offs(3),offs(4));
                Fun_AFG_RIGOL(vFG_R1,signal_n(5,:),signal_n(6,:),am(5),am(6),offs(5),offs(6));
                Fun_AFG_RIGOL(vFG_R2,signal_n(7,:),signal_n(8,:),am(7),am(8),offs(7),offs(8));
                
                %fclose(vFG);
                pause(0.2);

                % read spectrum from SA 
                % for S21
                line0=1;line1=0;
                USB6525(sUSB,line0,line1);
                spec21=Fun_SA(vSA,num_points);                
                As21=spec21(peakposition);
                
                pause(0.2);
                
               % line0=0;line1=1;
               % USB6525(sUSB,line0,line1);
               % spec11=Fun_SA(vSA,num_points);
               % As11=spec11(peakposition);
                
                As21_linear=10.^(As21./10);
               %As11_linear=10.^((As11+C_11)./10);
                 
                %figure(1)
                %plot(spec);hold on;
                %plot(peakposition,As21,'o');
                
                %%%% objective function define the order of harmonic that requires maximum
                %%%% efficiency. -As11_linear(nmax+1) is to substract 0 order
                %%%%  if we only care sideband directionality
                              
                % Record the spectrum
                    As21_Gn(n_geno,n_trans,:)=As21;   % peak values
                    %As11_Gn(n_geno,:)=As11;
                    As21_linear_Gn(n_geno,n_trans,:)=As21_linear; 

                    spec21_Gn(n_geno,n_trans,:)=spec21;  % raw spectrum
                    %spec21_Gn(n_geno,:)=spec21;
                 
                end
                
                
            end
            
                for n_geno=1:N_Geno
                OBJ1(n_geno)=As21_linear_Gn(n_geno,1,n_objhm)./As21_linear_Gn(n_geno,1,n_objhm+2);   
                OBJ2(n_geno)=(As21_linear_Gn(n_geno,1,n_objhm)+1.*As21_linear_Gn(n_geno,1,n_objhm+2))./(sum(sum(As21_linear_Gn(n_geno,:,:)))-As21_linear_Gn(n_geno,2,n_objhm+1)); 
                
                OBJ3(n_geno,:)=As21_linear_Gn(n_geno,1,n_objhm);
                
                OBJ4(n_geno)=(As21_linear_Gn(n_geno,2,n_objhm)+1.*As21_linear_Gn(n_geno,2,n_objhm+2))./sum(sum(As21_linear_Gn(n_geno,:,:))); 
                
                OBJ(n_geno)=min([OBJ1(n_geno),1/OBJ1(n_geno)]).*OBJ2(n_geno);%*OBJ3(n_geno);
                end
                 

            % sort the objective function        
            [Sobj,Cn]=sort(OBJ);

            %%% Sort the parameter according the objective function achieved
            SortA=A(:,:,Cn); 
            SortB=B(:,:,Cn);
            SortPH=PH(:,:,Cn); 
            SortAM=AM(:,:,Cn); 
            SortOF=OF(:,:,Cn); 


            % record the best spectrum for each generation 
            BestOBJ(n_gr)=Sobj(end);
            Best_As21(n_gr,:,:)=As21_Gn(Cn(end),:,:);
            %Best_As11(n_gr,:)=As11_Gn(Cn(end),:);

            Best_spec21(n_gr,:,:)=spec21_Gn(Cn(end),:,:);
            %Best_spec11(n_gr,:)=spec11_Gn(Cn(end),:);

            % record the best parameter for each generation
            Best_A(:,:,n_gr)=SortA(:,:,end);
            Best_B(:,:,n_gr)=SortB(:,:,end);
            Best_PH(:,:,n_gr)=SortPH(:,:,end);
            Best_AM(:,:,n_gr)=SortAM(:,:,end);
            Best_OF(:,:,n_gr)=SortOF(:,:,end);

            %%% Choose the best 6 parameters for the current generation
            BestA(:,:,1:6)=SortA(:,:,end-5:end); 
            BestB(:,:,1:6)=SortB(:,:,end-5:end);
            BestPH(:,:,1:6)=SortPH(:,:,end-5:end);
            BestAM(:,:,1:6)=SortAM(:,:,end-5:end);
            BestOF(:,:,1:6)=SortOF(:,:,end-5:end);

            %%% Generate new generation via hybridization

            % for 6 best chosen genos, they produce C_6,5 /2 =15 new genos in
            % the next generation
        
                n=1;
                for ii=1:5

                    for jj=ii+1:6

                NA(:,:,n)=(BestA(:,:,ii)+BestA(:,:,jj))/2;
                NB(:,:,n)=(BestB(:,:,ii)+BestB(:,:,jj))/2;
                NPH(:,:,n)=(BestPH(:,:,ii)+BestPH(:,:,jj))/2;
                NAM(:,:,n)=(BestAM(:,:,ii)+BestAM(:,:,jj))/2; 
                NOF(:,:,n)=(BestOF(:,:,ii)+BestOF(:,:,jj))/2; 

                n=n+1;
                    end

                end


                NA(:,:,16:end)=SortA(:,:,16:end);
                NB(:,:,16:end)=SortB(:,:,16:end);
                NPH(:,:,16:end)=SortPH(:,:,16:end);
                NAM(:,:,16:end)=SortAM(:,:,16:end);
                NOF(:,:,16:end)=SortOF(:,:,16:end);


                if n_gr>4 

                    % sort all the best geno achieved in each generation so far
                    [S_bestobj,Cnbest]=sort(BestOBJ);

                    Bestsofar_A=Best_A(:,:,Cnbest);
                    Bestsofar_B=Best_B(:,:,Cnbest);
                    Bestsofar_PH=Best_PH(:,:,Cnbest);
                    Bestsofar_AM=Best_AM(:,:,Cnbest);
                    Bestsofar_OF=Best_OF(:,:,Cnbest);


                    % choose the best 4 as source of mutation for the new
                    % generation, this can avoid degration if objective function
                    % becomes worse as the system evolves

                NA(:,:,16:19)=  Bestsofar_A(:,:,end-3:end);
                NB(:,:,16:19)=  Bestsofar_B(:,:,end-3:end);
                NPH(:,:,16:19)=  Bestsofar_PH(:,:,end-3:end);
                NAM(:,:,16:19)=  Bestsofar_AM(:,:,end-3:end);
                NOF(:,:,16:19)=  Bestsofar_OF(:,:,end-3:end);

                end                      



                if pn>1

                NA(:,:,20:20+pn-2)=A_best_allpass(:,:,1:pn-1);
                NB(:,:,20:20+pn-2)=B_best_allpass(:,:,1:pn-1);
                NPH(:,:,20:20+pn-2)=PH_best_allpass(:,:,1:pn-1);
                NAM(:,:,20:20+pn-2)=AM_best_allpass(:,:,1:pn-1);
                NOF(:,:,20:20+pn-2)=OF_best_allpass(:,:,1:pn-1);

                end
        
           
           
        % draw the best waveform and spectrum
                    
              a=Best_A(:,:,n_gr);
              b=Best_B(:,:,n_gr);
              ph=Best_PH(:,:,n_gr);
              am=Best_AM(:,:,n_gr);
              offs=Best_OF(:,:,n_gr);
              
              if flag_idenwf==1

                    a(2:2:end,:)=a(1:2:end,:); 
                    b(2:2:end,:)=b(1:2:end,:); 
                    
              end
              
              
                for nn=1:n_ch
                    
                    for n=1:n_od 
                        
                    sign_od(nn,n,:)=a(nn,n).*sin(n.*(2*pi.*Freq.*t+ph(nn,1)))+b(nn,n).*cos(n.*(2*pi.*Freq.*t+ph(nn,1)));  

                    end
                
                    signal(nn,:)=sum(sign_od(nn,:,:));
                    signal_n(nn,:)=signal(nn,:)./max(abs(signal(nn,:)));
                   
                    V_best(nn,:)=offs(nn)+0.5*am(nn).*signal_n(nn,:);
                end
                
                
        
                figure(110)
                subplot(1,3,1)
                plot(t,V_best(1,:),'r');hold on;
                plot(t,V_best(2,:),'b');hold on;
                plot(t,V_best(3,:),':c');hold on;
                plot(t,V_best(4,:),':m');hold off;
                
                legend('V_{A,1}','V_{A,2}','V_{B,1}','V_{B,2}')

                subplot(1,3,2)                
                Bst_As21(:,:)=Best_As21(n_gr,1,:);
                stem(Bst_As21,'*') %%% forward spectrum
                grid on
                title('forward spectrum')
                
                subplot(1,3,3)
                Bst_As11(:,:)=Best_As21(n_gr,2,:);
                stem(Bst_As11,'*') %%% backward spectrum
                grid on
                title('backward spectrum')
                
                figure (111)
                plot(BestOBJ,'-o'); 
        
           
        
        end

    %%
    [MM,bn]=max(BestOBJ);

    BestOBJ_allpass(pn,:)=MM;
    AS21_best_allpass(pn,:,:)=Best_As21(bn,:,:);
    
    spec21_best_allpass(pn,:,:)=Best_spec21(bn,:,:);    
    
    
    A_best_allpass(:,:,pn)=Best_A(:,:,bn);
    B_best_allpass(:,:,pn)=Best_B(:,:,bn);
    PH_best_allpass(:,:,pn)=Best_PH(:,:,bn);
    AM_best_allpass(:,:,pn)=Best_AM(:,:,bn);
    OF_best_allpass(:,:,pn)=Best_OF(:,:,bn);
    
    
    % draw the best waveform of all pass
            
              a=A_best_allpass(:,:,pn);
              b=B_best_allpass(:,:,pn);
              ph=PH_best_allpass(:,:,pn);
              am=AM_best_allpass(:,:,pn);
              offs=OF_best_allpass(:,:,pn);
              
              if flag_idenwf==1

                    a(2:2:end,:)=a(1:2:end,:); 
                    b(2:2:end,:)=b(1:2:end,:); 
                    
              end
              
              
                for nn=1:n_ch
                    
                    for n=1:n_od 
                        
                    sign_od(nn,n,:)=a(nn,n).*sin(n.*(2*pi.*Freq.*t+ph(nn,1)))+b(nn,n).*cos(n.*(2*pi.*Freq.*t+ph(nn,1)));  

                    end
                
                    signal(nn,:)=sum(sign_od(nn,:,:));
                    signal_n(nn,:)=signal(nn,:)./max(abs(signal(nn,:)));
                   
                    V_allpass(nn,:)=offs(nn)+0.5*am(nn).*signal_n(nn,:);
                end
                
               
       
              
                
                
    end

    
    %%
    
    BestOBJ_allpass_cwfr(:,:,n_cw)=BestOBJ_allpass;
    
    AS21_best_allpass_cwfr(:,:,:,n_cw)=AS21_best_allpass;
    
    spec21_best_allpass_cwfr(:,:,:,n_cw)=spec21_best_allpass;
     
    A_best_allpass_cwfr(:,:,:,n_cw)=A_best_allpass;
    B_best_allpass_cwfr(:,:,:,n_cw)=B_best_allpass;
    PH_best_allpass_cwfr(:,:,:,n_cw)=PH_best_allpass;    
    AM_best_allpass_cwfr(:,:,:,n_cw)=AM_best_allpass;
    OF_best_allpass_cwfr(:,:,:,n_cw)=OF_best_allpass;
    
    
    V_allpass_cwfr(:,:,n_cw)=V_allpass;
    
    
    % Choose the best waveform and change the relative phase 
    [cm,ci]=max(BestOBJ_allpass);
    
    a=A_best_allpass(:,:,ci);
    b=B_best_allpass(:,:,ci);
    ph=PH_best_allpass(:,:,ci);    
    am=AM_best_allpass(:,:,ci);
    offs=OF_best_allpass(:,:,ci);
    
    
    
   %%%%%%%%%%%%% plot the best waveform and spectra for all passes
       figure(999)
       BestFWS(:,:)=AS21_best_allpass(ci,1,:);
       BestBWS(:,:)=AS21_best_allpass(ci,2,:);
   
               if flag_idenwf==1

                    a(2:2:end,:)=a(1:2:end,:); 
                    b(2:2:end,:)=b(1:2:end,:); 
                    
              end
              
              
                for nn=1:n_ch
                    
                    for n=1:n_od 
                        
                    sign_od(nn,n,:)=a(nn,n).*sin(n.*(2*pi.*Freq.*t+ph(nn,1)))+b(nn,n).*cos(n.*(2*pi.*Freq.*t+ph(nn,1)));  

                    end
                
                    signal(nn,:)=sum(sign_od(nn,:,:));
                    signal_n(nn,:)=signal(nn,:)./max(abs(signal(nn,:)));
                   
                    V_allpass(nn,:)=offs(nn)+0.5*am(nn).*signal_n(nn,:);
                end
                
                
     % remeasure the best spectrum       

                % write signal to AFG
                Fun_AFG2(vFG2,signal_n(1,:),signal_n(2,:),am(1),am(2),offs(1),offs(2));
                Fun_AFG(vFG,signal_n(3,:),signal_n(4,:),am(3),am(4),offs(3),offs(4));
                Fun_AFG_RIGOL(vFG_R1,signal_n(5,:),signal_n(6,:),am(5),am(6),offs(5),offs(6));
                Fun_AFG_RIGOL(vFG_R2,signal_n(7,:),signal_n(8,:),am(7),am(8),offs(7),offs(8));

                %Fun_AFG2(vFG2,signal_n(2,:),signal_n(1,:),am(2),am(1),offs(2),offs(1));
                %Fun_AFG(vFG,signal_n(4,:),signal_n(3,:),am(4),am(3),offs(4),offs(3));
                %Fun_AFG_RIGOL(vFG_R1,signal_n(6,:),signal_n(5,:),am(6),am(5),offs(6),offs(5));
                %Fun_AFG_RIGOL(vFG_R2,signal_n(8,:),signal_n(7,:),am(8),am(7),offs(8),offs(7));

                %fclose(vFG);
                pause(0.2);
        
        % measure forward and backward signals
                for n_trans=1:2
                x=X(n_trans);y=Y(n_trans);
                [pos_x,pos_y]=Fun_scan_TS(vTS,x,y); 
                
                spec21=Fun_SA(vSA,num_points); 
                
                As21_re(:,n_trans)=spec21(peakposition);

                end

                As21_re_linear=10.^(As21_re./10);
                
            
   
        subplot(1,3,1)
        plot(t,V_allpass(1,:),'r');hold on;
        plot(t,V_allpass(2,:),'b');hold on;
        plot(t,V_allpass(3,:),':c');hold on;
        plot(t,V_allpass(4,:),':m');hold off;

        subplot(1,3,2)
        stem(BestFWS,'*');hold on;
        stem(As21_re(:,1),'ro');hold off;
        
        title('Forward scattering_{best of all pass}')

        subplot(1,3,3)
        stem(BestBWS,'*');hold on;
        stem(As21_re(:,2),'ro');hold off;
        title('Backward scattering_{best of all pass}')
   
    
        
        
        
end

%%
%%%%%% do a circle scan using the best waveform

    [cm,ci]=max(BestOBJ_allpass);
    
    a=A_best_allpass(:,:,ci);
    b=B_best_allpass(:,:,ci);
    ph=PH_best_allpass(:,:,ci);    
    am=AM_best_allpass(:,:,ci);
    offs=OF_best_allpass(:,:,ci);
    
    dph=1.*[0,0,pi/2,pi/2,pi,pi,3*pi/2,3*pi/2].';
    
   %%% contruct signals
            for nn=1:n_ch

                for n=1:n_od 

                sign_od(nn,n,:)=a(nn,n).*sin(n.*(2*pi.*Freq.*t+ph(nn,1)+dph(nn,1)))+b(nn,n).*cos(n.*(2*pi.*Freq.*t+ph(nn,1)+dph(nn,1)));  

                end

                signal(nn,:)=sum(sign_od(nn,:,:));
                signal_n(nn,:)=signal(nn,:)./max(abs(signal(nn,:)));

            end


        % 4 pairs
        % write signal to AFG                
        Fun_AFG2(vFG2,signal_n(1,:),signal_n(2,:),am(1),am(2),offs(1),offs(2));
        Fun_AFG(vFG,signal_n(3,:),signal_n(4,:),am(3),am(4),offs(3),offs(4));
        Fun_AFG_RIGOL(vFG_R1,signal_n(5,:),signal_n(6,:),am(5),am(6),offs(5),offs(6));
        Fun_AFG_RIGOL(vFG_R2,signal_n(7,:),signal_n(8,:),am(7),am(8),offs(7),offs(8));

        %fclose(vFG);
        pause(0.2);

%
        %%% parameters of translational stage
            THETA_scan=(0:2.5:360).*pi/180;
            R_scan=150;
            X_scan=R_scan.*cos(THETA_scan+pi/4)+20;
            Y_scan=R_scan.*sin(THETA_scan+pi/4)+10;
            N_scan =length(X_scan);
            As21_scan_1=zeros(length(peakposition),N_scan);
            
        for n_scan=1:1:N_scan
                
                x=X_scan(n_scan);y=Y_scan(n_scan);
                [pos_x,pos_y]=Fun_scan_TS(vTS,x,y);

            spec21=Fun_SA(vSA,num_points);                
            As21_scan_1(:,n_scan)=spec21(peakposition);

            pause(0.2);
          
        end

          As21_scan_linear=10.^(As21_scan_1./10);
        % 
        figure(301)
        subplot(1,3,1)
        polar(THETA_scan,As21_scan_linear(n_objhm+1,:),'*-r');hold on
        title('0th order')
        
        subplot(1,3,2)
        polar(THETA_scan,As21_scan_linear(n_objhm,:),'*-k');hold on;
        title('-1st order')
        
        subplot(1,3,3)
        polar(THETA_scan,As21_scan_linear(n_objhm+2,:),'*-k');hold on;
        title('1st order')
%%
n=5;
%As21_scan_beamsteering1(:,:,1)=As21_scan_beamsteering3(:,:,1);
As21_scan_beamsteering1(:,:,n)=As21_scan_1;
%phase_beamsteering(:,n)=dph;

%%
As21_scan_2_1st(:,:)=10.^(As21_scan_beamsteering2(n_objhm+2,:,:)/10);

figure(3)
 polar(THETA_scan,As21_scan_2_1st(:,5).','*-r');hold on;
        title('1st order')



%% do a matrix scan

            X_matrix=[-80:2:150]+20;
            Y_matrix=[-80:2:150]+10;
            
            N_scan =length(X_matrix);
            As21_scan_m=zeros(length(peakposition),N_scan);
            
        for n_scanx=1:1:N_scan
            
            for n_scany=1:1:N_scan
                
                x=X_matrix(n_scanx);y=Y_matrix(n_scany);
                [pos_x,pos_y]=Fun_scan_TS(vTS,x,y);

            spec21=Fun_SA(vSA,num_points);                
            As21_scan_m3(:,n_scanx,n_scany)=spec21(peakposition);

            pause(0.2);
          
            end
        end

          As21_scan_linear_m3=10.^(As21_scan_m3./10);
          
        
          As21_scan_0th_m3(:,:)=As21_scan_linear_m3(n_objhm+1,:,:);
          As21_scan_p1_m3(:,:)=As21_scan_linear_m3(n_objhm+2,:,:);
          As21_scan_n1_m3(:,:)=As21_scan_linear_m3(n_objhm,:,:);
          
          %%
          figure(204)
          subplot(1,3,1)
          sf=surf(X_matrix,Y_matrix,As21_scan_0th_m3);
          set(sf, 'EdgeColor', 'none');
          view(0,90);
          axis([-60,150,-70,150])
          title('0th order')
          caxis([0,1e-6])
          
          
          subplot(1,3,2)
          sf=surf(X_matrix,Y_matrix,As21_scan_p1_m3);
          set(sf, 'EdgeColor', 'none');
          view(0,90);
          axis([-60,150,-70,150])
          title('1st order')
          caxis([0,1e-7])
          
          subplot(1,3,3)
          sf=surf(X_matrix,Y_matrix,As21_scan_n1_m3);
          set(sf, 'EdgeColor', 'none');
          view(0,90);
          axis([-60,150,-70,150])
          title('-1st order')
          caxis([0,1e-7])
          
          
          %% matrix scan in IQ mode
          
            X_matrix=[-80:5:80]+20;
            Y_matrix=[-80:5:80]+10;
            
            N_scan =length(X_matrix);
            As21_scan_m=zeros(length(peakposition),N_scan);
            
        for n_scanx=1:1:N_scan
            
            for n_scany=1:1:N_scan
                
                x=X_matrix(n_scanx);y=Y_matrix(n_scany);
                [pos_x,pos_y]=Fun_scan_TS(vTS,x,y);

            fwrite(vSA, 'INIT:CONT OFF'); 

            % perform sweep
            tic
            fwrite(vSA, 'INIT;');
            % wait for sweep to complete
            fprintf(vSA, '*OPC?');

            fprintf(vSA, 'TRACE? TRACE1');
            pause(0.5);

            read_values = fscanf(vSA,'%f,', 2.*num_points);

            spec_real(:,:)=read_values(1:num_points);
            spec_imag(:,:)=read_values(1+num_points:2*num_points);

            spec_time(:,n_scanx,n_scany)=spec_real+1i.*spec_imag;
            
            end
            
        end
        

%% extract the data

scan_2d_grd1_ph1_n1(:,1)=10.^(As21_scan_beamsteering1(n_objhm,:,1)./10);
scan_2d_grd1_ph2_n1(:,1)=10.^(As21_scan_beamsteering1(n_objhm,:,2)./10);
scan_2d_grd1_ph3_n1(:,1)=10.^(As21_scan_beamsteering1(n_objhm,:,3)./10);
scan_2d_grd1_ph4_n1(:,1)=10.^(As21_scan_beamsteering1(n_objhm,:,4)./10);
scan_2d_grd1_ph5_n1(:,1)=10.^(As21_scan_beamsteering1(n_objhm,:,5)./10);
scan_2d_grd1_ph6_n1(:,1)=10.^(As21_scan_beamsteering1(n_objhm,:,6)./10);

scan_2d_grd2_ph1_n1(:,1)=10.^(As21_scan_beamsteering2(n_objhm,:,1)./10);
scan_2d_grd2_ph2_n1(:,1)=10.^(As21_scan_beamsteering2(n_objhm,:,2)./10);
scan_2d_grd2_ph3_n1(:,1)=10.^(As21_scan_beamsteering2(n_objhm,:,3)./10);
scan_2d_grd2_ph4_n1(:,1)=10.^(As21_scan_beamsteering2(n_objhm,:,4)./10);
scan_2d_grd2_ph5_n1(:,1)=10.^(As21_scan_beamsteering2(n_objhm,:,5)./10);
scan_2d_grd2_ph6_n1(:,1)=10.^(As21_scan_beamsteering2(n_objhm,:,6)./10);

scan_2d_grd3_ph1_n1(:,1)=10.^(As21_scan_beamsteering3(n_objhm,:,1)./10);
scan_2d_grd3_ph2_n1(:,1)=10.^(As21_scan_beamsteering3(n_objhm,:,2)./10);
scan_2d_grd3_ph3_n1(:,1)=10.^(As21_scan_beamsteering3(n_objhm,:,3)./10);
scan_2d_grd3_ph4_n1(:,1)=10.^(As21_scan_beamsteering3(n_objhm,:,4)./10);
scan_2d_grd3_ph5_n1(:,1)=10.^(As21_scan_beamsteering3(n_objhm,:,5)./10);
scan_2d_grd3_ph6_n1(:,1)=10.^(As21_scan_beamsteering3(n_objhm,:,6)./10);

scan_2d_grd5_ph2_n1(:,1)=10.^(As21_scan_beamsteering5(n_objhm,:,2)./10);
scan_2d_grd5_ph3_n1(:,1)=10.^(As21_scan_beamsteering5(n_objhm,:,3)./10);

scan_2d_grd1_ph1_n1=[scan_2d_grd1_ph1_n1(109:144);scan_2d_grd1_ph1_n1(1:109)];
scan_2d_grd1_ph2_n1=[scan_2d_grd1_ph2_n1(109:144);scan_2d_grd1_ph2_n1(1:109)];
scan_2d_grd1_ph3_n1=[scan_2d_grd1_ph3_n1(109:144);scan_2d_grd1_ph3_n1(1:109)];
scan_2d_grd1_ph4_n1=[scan_2d_grd1_ph4_n1(109:144);scan_2d_grd1_ph4_n1(1:109)];
scan_2d_grd1_ph5_n1=[scan_2d_grd1_ph5_n1(109:144);scan_2d_grd1_ph5_n1(1:109)];
scan_2d_grd1_ph6_n1=[scan_2d_grd1_ph6_n1(109:144);scan_2d_grd1_ph6_n1(1:109)];

scan_2d_grd2_ph1_n1=[scan_2d_grd2_ph1_n1(109:144);scan_2d_grd2_ph1_n1(1:109)];
scan_2d_grd2_ph2_n1=[scan_2d_grd2_ph2_n1(109:144);scan_2d_grd2_ph2_n1(1:109)];
scan_2d_grd2_ph3_n1=[scan_2d_grd2_ph3_n1(109:144);scan_2d_grd2_ph3_n1(1:109)];
scan_2d_grd2_ph4_n1=[scan_2d_grd2_ph4_n1(109:144);scan_2d_grd2_ph4_n1(1:109)];
scan_2d_grd2_ph5_n1=[scan_2d_grd2_ph5_n1(109:144);scan_2d_grd2_ph5_n1(1:109)];
scan_2d_grd2_ph6_n1=[scan_2d_grd2_ph6_n1(109:144);scan_2d_grd2_ph6_n1(1:109)];

scan_2d_grd3_ph1_n1=[scan_2d_grd3_ph1_n1(109:144);scan_2d_grd3_ph1_n1(1:109)];
scan_2d_grd3_ph2_n1=[scan_2d_grd3_ph2_n1(109:144);scan_2d_grd3_ph2_n1(1:109)];
scan_2d_grd3_ph3_n1=[scan_2d_grd3_ph3_n1(109:144);scan_2d_grd3_ph3_n1(1:109)];
scan_2d_grd3_ph4_n1=[scan_2d_grd3_ph4_n1(109:144);scan_2d_grd3_ph4_n1(1:109)];
scan_2d_grd3_ph5_n1=[scan_2d_grd3_ph5_n1(109:144);scan_2d_grd3_ph5_n1(1:109)];
scan_2d_grd3_ph6_n1=[scan_2d_grd3_ph6_n1(109:144);scan_2d_grd3_ph6_n1(1:109)];

%%
scan_2d_grd5_ph2_n1=[scan_2d_grd5_ph2_n1(109:144);scan_2d_grd5_ph2_n1(1:109)];
scan_2d_grd5_ph3_n1=[scan_2d_grd5_ph3_n1(109:144);scan_2d_grd5_ph3_n1(1:109)];


theta_scan=[90:-2.5:0,360-(2.5:2.5:270)].';

%%
save theta_scan.txt -ascii theta_scan

save scan_2d_grd1_ph1_n1.txt -ascii scan_2d_grd1_ph1_n1
save scan_2d_grd1_ph2_n1.txt -ascii scan_2d_grd1_ph2_n1
save scan_2d_grd1_ph3_n1.txt -ascii scan_2d_grd1_ph3_n1
save scan_2d_grd1_ph4_n1.txt -ascii scan_2d_grd1_ph4_n1
save scan_2d_grd1_ph5_n1.txt -ascii scan_2d_grd1_ph5_n1
save scan_2d_grd1_ph6_n1.txt -ascii scan_2d_grd1_ph6_n1

save scan_2d_grd2_ph1_n1.txt -ascii scan_2d_grd2_ph1_n1
save scan_2d_grd2_ph2_n1.txt -ascii scan_2d_grd2_ph2_n1
save scan_2d_grd2_ph3_n1.txt -ascii scan_2d_grd2_ph3_n1
save scan_2d_grd2_ph4_n1.txt -ascii scan_2d_grd2_ph4_n1
save scan_2d_grd2_ph5_n1.txt -ascii scan_2d_grd2_ph5_n1
save scan_2d_grd2_ph6_n1.txt -ascii scan_2d_grd2_ph6_n1

save scan_2d_grd3_ph1_n1.txt -ascii scan_2d_grd3_ph1_n1
save scan_2d_grd3_ph2_n1.txt -ascii scan_2d_grd3_ph2_n1
save scan_2d_grd3_ph3_n1.txt -ascii scan_2d_grd3_ph3_n1
save scan_2d_grd3_ph4_n1.txt -ascii scan_2d_grd3_ph4_n1
save scan_2d_grd3_ph5_n1.txt -ascii scan_2d_grd3_ph5_n1
save scan_2d_grd3_ph6_n1.txt -ascii scan_2d_grd3_ph6_n1

save scan_2d_grd5_ph2_n1.txt -ascii scan_2d_grd5_ph2_n1
save scan_2d_grd5_ph3_n1.txt -ascii scan_2d_grd5_ph3_n1
%%
As21_scan_linear_m3_p1(:,:)=As21_scan_linear_m3(n_objhm+2,:,:);
As21_scan_linear_m3_n1(:,:)=As21_scan_linear_m3(n_objhm,:,:);

save As21_scan_linear_G3_p1.txt -ascii As21_scan_linear_m3_p1
save As21_scan_linear_G3_n1.txt -ascii As21_scan_linear_m3_n1

%%
As21_scan_linear_lens_p1(:,:)=As21_scan_linear_lens2(n_objhm+2,:,:);
As21_scan_linear_lens_n1(:,:)=As21_scan_linear_lens2(n_objhm,:,:);

save As21_scan_linear_G5_p1.txt -ascii As21_scan_linear_lens_p1
save As21_scan_linear_G5_n1.txt -ascii As21_scan_linear_lens_n1

%%
figure(111)
subplot(2,2,1)
sf=surf(X_matrix,Y_matrix,As21_scan_linear_lens_n1);
set(sf, 'EdgeColor', 'none');
view(0,90);
axis([-60,150,-70,150])
title('-1st order')
caxis([0,5e-7])

subplot(2,2,2)
sf=surf(X_matrix,Y_matrix,As21_scan_linear_lens_p1);
set(sf, 'EdgeColor', 'none');
view(0,90);
axis([-60,150,-70,150])
title('1st order')
caxis([0,5e-7])

subplot(2,2,3)
diffAs21_n1=diff(sign(diff(As21_scan_linear_lens_n1)));
sf=surf(diffAs21_n1);
set(sf, 'EdgeColor', 'none');
view(0,90);
%axis([-60,150,-70,150])
title('1st order')

subplot(2,2,4)
diffAs21_p1=diff(sign(diff(As21_scan_linear_lens_p1)));
sf=surf(diffAs21_p1);
set(sf, 'EdgeColor', 'none');
view(0,90);
%axis([-60,150,-70,150])
title('1st order')
%caxis([0,5e-7])

iden_diff=diffAs21_p1;
figure(20)
sf=surf(iden_diff);
set(sf, 'EdgeColor', 'none');
view(0,90);
%axis([-60,150,-70,150])
title('1st order')

iden_diff(115,:)=iden_diff(114,:);
iden_diff(116,:)=iden_diff(114,:);

absiden_diff=0.25.*(4-abs(iden_diff));


%%
As21_scan_linear_lens_p1_NaN=As21_scan_linear_lens_p1.*absiden_diff;
As21_scan_linear_lens_p1_NaN(absiden_diff==0)=NaN;

%%
figure(112)
subplot(2,2,1)
sf=surf(X_matrix,Y_matrix,As21_scan_linear_lens_p1_NaN);
set(sf, 'EdgeColor', 'none');
view(0,90);
axis([-60,150,-70,150])
title('-1st order')

caxis([0,5e-7])

subplot(2,2,2)
sf=surf(X_matrix,Y_matrix,As21_scan_linear_lens_p1);
set(sf, 'EdgeColor', 'none');
view(0,90);
axis([-60,150,-70,150])
title('1st order')
caxis([0,5e-7])

%%
figure(21)
sf=surf(As21_scan_linear_lens_p1_remove);
set(sf, 'EdgeColor', 'none');
view(0,90);
%axis([-60,150,-70,150])
title('1st order')

%%
X_matrix_i=min(X_matrix):0.2:max(X_matrix);
Y_matrix_i=min(Y_matrix):0.2:max(Y_matrix);

[X_matrix_I,Y_matrix_I]=meshgrid(X_matrix_i,Y_matrix_i);

As21_scan_linear_lens_p1_I=interp2(X_matrix,Y_matrix,As21_scan_linear_lens_p1_remove,X_matrix_I,Y_matrix_I);
As21_scan_linear_lens_n1_I=interp2(X_matrix,Y_matrix,As21_scan_linear_lens_n1,X_matrix_I,Y_matrix_I);

%%
figure(112)
subplot(1,2,1)
sf=surf(X_matrix_I,Y_matrix_I,As21_scan_linear_lens_n1_I);
set(sf, 'EdgeColor', 'none');
view(0,90);
axis([-60,150,-70,150])
title('-1st order')
caxis([0,5e-7])

subplot(1,2,2)
sf=surf(X_matrix_I,Y_matrix_I,As21_scan_linear_lens_p1_I);
set(sf, 'EdgeColor', 'none');
view(0,90);
axis([-60,150,-70,150])
title('1st order')
caxis([0,5e-7])