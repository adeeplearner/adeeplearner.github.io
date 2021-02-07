function [x, mterm] = gradient_descent_momentum(x, delx, delxt1, lr, alpha)
mterm = alpha * delx + (1-alpha) * delxt1;
x = x - lr * mterm;
end

