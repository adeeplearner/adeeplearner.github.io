ffmpeg -i compressed_pbola_gradient_descent.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" compressed_pbola_gradient_descent.mp4

ffmpeg -i pbola_gradient_descent.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" pbola_gradient_descent.mp4

#ffmpeg -i video.mp4 -ss 00:00:08 -t 00:00:14 -async 1 cut.mp4
