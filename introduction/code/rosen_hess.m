function H = rosen_hess(x)
    H = zeros(2,2);
    H(1,1) = 2+400*(3*x(1)^2-x(2));
    H(2,2) = 200;
    H(1,2) = -400*x(1);
    H(2,1) = H(1,2);
end
