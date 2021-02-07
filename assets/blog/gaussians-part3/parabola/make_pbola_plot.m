function make_pbola_plot(x, y, X, Y, Z)

subplot(1, 2, 1);
surf(X,Y,Z);
hold on;
plot3(x, y, pbola(x, y)+8, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 15);
xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis');
view([45 50]);
% view([90 90]);
hold off;

subplot(1, 2, 2);
surf(X,Y,Z);
hold on;
plot3(x, y, pbola(x, y)+8, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 15);
xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis');
%view([45 40]);
% view([90 90]);
view(2);
hold off;

set(gcf, 'Renderer', 'painters', 'Position', [10 10 900 300])

end

