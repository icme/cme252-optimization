function [f g H] = rosen(x)

% compute objective function
f = (1-x(1))^2 + 100*(x(2)-x(1)^2)^2;

%(1-x).^2 + 100*(y-x.^2).^2;

% compute gradient
if nargout > 1
    g = [0;0];
    g(1) = -2*(1-x(1)) - 400*x(1)*(x(2)-x(1)^2);
    g(2) = 200*(x(2)-x(1)^2);
end

% compute hessian
if nargout > 2
    H = zeros(2,2);
    H(1,1) = 2+400*(3*x(1)^2-x(2));
    H(2,2) = 200;
    H(1,2) = -400*x(1);
    H(2,1) = H(1,2);
end
    
end
