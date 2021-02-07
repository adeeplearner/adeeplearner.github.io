convert   -delay 10  -loop 0  *.png -crop 1182x451+92+18 +repage -resize 50%  pbola_gradient_descent.gif
convert pbola_gradient_descent.gif -coalesce  -fuzz 2% +dither -layers Optimize +map compressed_pbola_gradient_descent.gif
