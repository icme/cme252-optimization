function rosen_nm

% newton's method on rosenbrock function

options = optimoptions('fminunc');
options.Algorithm = 'trust-region'
options.Hessian = 'on';
options.MaxFunEvals = 4000;
options.MaxIter = 4000;
%options.DerivativeCheck = 'on';
options.GradObj = 'on';
%options.PlotFcns = @optimplotfval;

i = 1;
xi = zeros(2,0);
fni = zeros(1,0);
cvi = zeros(1,0);

    function stop = xcollect(x,optimValues,state)
        %keyboard
        if strcmp(state,'iter')
            xi(:,i) = x;
            fni(1,i) = optimValues.funccount;
            cvi(1,i) = optimValues.firstorderopt;
            i = i + 1;
        end
        stop = false;
    end

options.OutputFcn = @xcollect;

[x,fval,exitflag,output] = fminunc(@rosen,[-1.25;0],options);

%x
%output

%rosen_contour
%plot(xi(1,:),xi(2,:))
%plot(xi(1,:),xi(2,:),'ko')

% save data
data_nm.x = x;
data_nm.xi = xi;
data_nm.fni = fni;
data_nm.cvi = cvi;
data_nm.output = output;

save('data_nm.mat','data_nm');

%keyboard

end