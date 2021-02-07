import glob
import os

from PIL import Image

outdir = 'cropped'
scale = 1.3
BBOX = [600, 176, 1417, 795]

files = glob.glob('./*.png')

for file in files:
    print('process %s' % file)
    inimage = Image.open(file).crop(BBOX)
    width, height = inimage.size
    inimage.resize((int(width/scale), int(height/scale)), Image.BILINEAR).save(os.path.join(outdir, os.path.basename(file)))


