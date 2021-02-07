convert   -delay 70   -loop 0   *.png  gradient_descent.gif
convert gradient_descent.gif -coalesce  -fuzz 2% +dither -layers Optimize +map compressed_gradient_descent.gif
rm gradient_descent.gif
