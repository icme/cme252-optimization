function rosen_iter

% gradient descent data
load data_gd

% newton method data
load data_nm

subplot(1,2,1)
plot_iter(data_gd.xi)
title('Gradient descent','fontsize',14)
axis([-.2 .2 -.2 .2])
subplot(1,2,2)
plot_iter(data_nm.xi)
title('Newton''s method','fontsize',14)

print('gd-nm-iter-2','-depsc2')

end

function plot_iter(xi)

x = linspace(-1.5,1.5);
y = linspace(-1,3);

[xx,yy] = meshgrid(x,y);

ff = zeros(length(x),length(y));

for i = 1:length(x)
    for j = 1:length(y)
        ff(j,i) = rosen([x(i);y(j)]);
    end
end

levels = 10:30:300;

contour(x,y,ff,levels,'linewidth',1)
axis([-1.5 1.5 -1 3])
axis square
hold on

plot(xi(1,:),xi(2,:),'ko-',...
    'linewidth',1.1,...
    'MarkerEdgeColor','r',...
    'markerfacecolor',[1,0,0])

end