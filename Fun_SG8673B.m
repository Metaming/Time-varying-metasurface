% This is a program to control signal generator HP-8673B
function Fun_SG8637B(vSG,cwfr,power_dbm)

outpw=['LE ',num2str(power_dbm),' DM'];
outfr=['FR',num2str(cwfr),'GZ'];
fwrite(vSG, outpw);
fwrite(vSG, outfr);
end

%fclose(vSG);