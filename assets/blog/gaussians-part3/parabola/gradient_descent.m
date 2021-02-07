function [x] = gradient_descent(x, delx, lr)
x = x - lr * delx;
end

