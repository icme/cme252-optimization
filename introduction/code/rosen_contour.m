function rosen_contour

x = linspace(-1.5,1.5);
y = linspace(-1,3);

[xx,yy] = meshgrid(x,y);

ff = zeros(length(x),length(y));

for i = 1:length(x)
    for j = 1:length(y)
        ff(j,i) = rosen([x(i);y(j)]);
    end
end

levels = 10:20:300;
figure, contour(x,y,ff,levels,'linewidth',2), colorbar
axis([-1.5 1.5 -1 3]), axis square, hold on

plot(1,1,'ko','markersize',9,...
    'markerfacecolor','r')

title('(1-x)^2 + 100*(y-x^2)^2','fontsize',14)
xlabel('x','fontsize',14)
ylabel('y','fontsize',14)

print('rosen-contour','-depsc2')

%keyboard

end