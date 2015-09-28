function rosen_conv

% gradient descent data
load data_gd

% newton method data
load data_nm

% plot
subplot(1,2,1)
ngd = 100;
semilogy(1:ngd,data_gd.cvi(1:ngd),'linewidth',2)
title('Gradient descent','fontsize',14)
xlabel('iteration','fontsize',14)
ylabel('optimality condition','fontsize',14)
ylim([1e-5 1e3])
axis square

subplot(1,2,2)
nnm = length(data_nm.cvi);
semilogy(1:nnm,data_nm.cvi(1:nnm),'linewidth',2)
title('Newton''s method','fontsize',14)
xlabel('iteration','fontsize',14)
ylabel('optimality condition','fontsize',14)
ylim([1e-5 1e3])
axis square

print('rosen-conv','-depsc2')

keyboard

end
