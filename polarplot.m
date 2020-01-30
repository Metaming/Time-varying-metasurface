

n_freq=12000-800+1;

angle2=([0:360/72:360]).*pi/180;

angle0=[0:360/72:360];
angleI=[0:1:360];

%figure(211);
%polar(angle2.',10.^(recorded_spectra_150c(:,n_freq)./10));hold on;

figure(215);
plot(angle2.',10.^(recorded_spectra_90c(:,n_freq)./10));hold on;


field0_0deg_n1=10.^(recorded_spectra_0c(:,n_freq)./10);
field0_30deg_n1=10.^(recorded_spectra_30c(:,n_freq)./10);
field0_60deg_n1=10.^(recorded_spectra_60c(:,n_freq)./10);
field0_90deg_n1=10.^(recorded_spectra_90c(:,n_freq)./10);

field0_0deg_p1=10.^(recorded_spectra_0c(:,n_freq+1600)./10);
field0_30deg_p1=10.^(recorded_spectra_30c(:,n_freq+1600)./10);
field0_60deg_p1=10.^(recorded_spectra_60c(:,n_freq+1600)./10);
field0_90deg_p1=10.^(recorded_spectra_90c(:,n_freq+1600)./10);

fieldI_0deg_n1=interp1(angle0,field0_0deg_n1,angleI,'cubic');
fieldI_30deg_n1=interp1(angle0,field0_30deg_n1,angleI,'cubic');
fieldI_60deg_n1=interp1(angle0,field0_60deg_n1,angleI,'cubic');
fieldI_90deg_n1=interp1(angle0,field0_90deg_n1,angleI,'cubic');
fieldI_0deg_p1=interp1(angle0,field0_0deg_p1,angleI,'cubic');
fieldI_30deg_p1=interp1(angle0,field0_30deg_p1,angleI,'cubic');
fieldI_60deg_p1=interp1(angle0,field0_60deg_p1,angleI,'cubic');
fieldI_90deg_p1=interp1(angle0,field0_90deg_p1,angleI,'cubic');

field_0deg_n1=fieldI_0deg_n1.'./max(fieldI_0deg_p1);
field_30deg_n1=fieldI_30deg_n1.'./max(fieldI_0deg_p1);
field_60deg_n1=fieldI_60deg_n1.'./max(fieldI_0deg_p1);
field_90deg_n1=fieldI_90deg_n1.'./max(fieldI_0deg_p1);

field_0deg_p1=fieldI_0deg_p1.'./max(fieldI_0deg_p1);
field_30deg_p1=fieldI_30deg_p1.'./max(fieldI_0deg_p1);
field_60deg_p1=fieldI_60deg_p1.'./max(fieldI_0deg_p1);
field_90deg_p1=fieldI_90deg_p1.'./max(fieldI_0deg_p1);

fieldn_0deg_n1=1/0.89.*[field_0deg_n1(45+180:361);field_0deg_n1(1:44+180)];
fieldn_30deg_n1=1/0.89.*[field_30deg_n1(45+180:361);field_30deg_n1(1:44+180)];
fieldn_60deg_n1=1/0.89.*[field_60deg_n1(45+180:361);field_60deg_n1(1:44+180)];
fieldn_90deg_n1=1/0.89.*[field_90deg_n1(45+180:361);field_90deg_n1(1:44+180)];
fieldn_0deg_p1=[field_0deg_p1(45+180:361);field_0deg_p1(1:44+180)];
fieldn_30deg_p1=[field_30deg_p1(45+180:361);field_30deg_p1(1:44+180)];
fieldn_60deg_p1=[field_60deg_p1(45+180:361);field_60deg_p1(1:44+180)];
fieldn_90deg_p1=[field_90deg_p1(45+180:361);field_90deg_p1(1:44+180)];


figure(215);
plot(angleI.',fieldn_0deg_n1,'k');hold on;
plot(angleI.',fieldn_0deg_p1,':k');hold on;
plot(angleI.',fieldn_30deg_n1,'b');hold on;
plot(angleI.',fieldn_60deg_n1,'r');hold on;
plot(angleI.',fieldn_90deg_n1,'g');hold on;
plot(angleI.',fieldn_30deg_p1,':b');hold on;
plot(angleI.',fieldn_60deg_p1,':r');hold on;
plot(angleI.',fieldn_90deg_p1,':g');hold off;



save scatterfield_2D_double_uni_0deg_n1.txt -ascii field_0deg_n1
save scatterfield_2D_double_uni_30deg_n1.txt -ascii field_30deg_n1
save scatterfield_2D_double_uni_60deg_n1.txt -ascii field_60deg_n1
save scatterfield_2D_double_uni_90deg_n1.txt -ascii field_90deg_n1

save scatterfield_2D_double_uni_0deg_p1.txt -ascii field_0deg_p1
save scatterfield_2D_double_uni_30deg_p1.txt -ascii field_30deg_p1
save scatterfield_2D_double_uni_60deg_p1.txt -ascii field_60deg_p1
save scatterfield_2D_double_uni_90deg_p1.txt -ascii field_90deg_p1