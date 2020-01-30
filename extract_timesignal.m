ABSS21_ave=(sum(ABSS21_test2.')).'./20;
PHS21_ave=(sum(PHS21_test2.')).'.*180/pi./20;
ABSS11_ave=(sum(ABSS11_test2.')).'./20;
PHS11_ave=(sum(PHS11_test2.')).'.*180/pi./20;

figure(21)
subplot(1,2,1)
plot(ABSS21_ave);hold on;
plot(ABSS11_ave);hold off;
subplot(1,2,2)
plot(PHS21_ave);hold on;
plot(PHS11_ave);hold off;

save s21_time_exp_double_uni.txt -ascii ABSS21_ave
save s21ph_time_exp_double_uni.txt -ascii PHS21_ave
save s11_time_exp_double_uni.txt -ascii ABSS11_ave
save s11ph_time_exp_double_uni.txt -ascii PHS11_ave
