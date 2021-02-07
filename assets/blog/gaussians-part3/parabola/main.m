clear all; close all; clf
% figure('Position', [10 10 800 600]);
% subplot(1, 2, 1); subplot(1, 2, 2);
% figure;

% get the surface which we will plot again and again
[X,Y] = meshgrid(-10:0.5:10,-10:0.5:10);
Z = pbola(X, Y);

save_every_epoch = 1;
epochs = 231;
save_fig = true;
% define a starting point
x=10;
y=4;

lr = 0.1;
alpha = 0.03;
delxt1=0;
delyt1=4;
for i=1:epochs
    clf;
    [delx, dely] = pbola_grad(x, y);

    [x, delxt1] = gradient_descent_momentum(x, delx, delxt1, lr, alpha);
    [y, delyt1] = gradient_descent_momentum(y, dely, delyt1, lr, alpha);
    make_pbola_plot(x, y, X, Y, Z);
    pause(0.1);
    
    if (mod(i-1, save_every_epoch) == 0) && save_fig 
        buffer = sprintf('saved_figures/%0.7d.png', i)
        saveas(gcf, buffer);
    end
    
    
end