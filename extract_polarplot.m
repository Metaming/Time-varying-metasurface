
theta=(0:1:359).';

NF=max(dfarfieldcst0deg(:,3));
P_ff_0=dfarfieldcst0deg(:,3)./NF;
P_ff_30=dfarfieldcst30deg(:,3)./NF;
P_ff_60=dfarfieldcst60deg(:,3)./NF;
P_ff_90=dfarfieldcst90deg(:,3)./NF;
P_ff_n30=dfarfieldcstn30deg(:,3)./NF;
P_ff_n60=dfarfieldcstn60deg(:,3)./NF;
P_ff_n90=dfarfieldcstn90deg(:,3)./NF;



figure(912);
polar(theta.*pi/180,P_ff_0);hold on;
polar(theta.*pi/180,P_ff_30);hold on;
polar(theta.*pi/180,P_ff_60);hold on;
polar(theta.*pi/180,P_ff_90);hold on;
polar(theta.*pi/180,P_ff_n30);hold on;
polar(theta.*pi/180,P_ff_n60);hold on;
polar(theta.*pi/180,P_ff_n90);hold off;

save P_2Dff_cst_0.txt -ascii P_ff_0
save P_2Dff_cst_30.txt -ascii P_ff_30
save P_2Dff_cst_60.txt -ascii P_ff_60
save P_2Dff_cst_90.txt -ascii P_ff_90
save P_2Dff_cst_n30.txt -ascii P_ff_n30
save P_2Dff_cst_n60.txt -ascii P_ff_n60
save P_2Dff_cst_n90.txt -ascii P_ff_n90