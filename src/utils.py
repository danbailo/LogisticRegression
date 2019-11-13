import glob
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 


def save_img(img_np, file_name, predict):
	img = Image.fromarray(img_np[:,:,:])

	draw = ImageDraw.Draw(img)
	fillcolor = "black"
	shadowcolor = "white"
	x, y = 1, 1

	text = "{:.3f}".format(predict)

	# thin border
	draw.text((x-1, y), text, fill=shadowcolor)
	draw.text((x+1, y), text, fill=shadowcolor)
	draw.text((x, y-1), text, fill=shadowcolor)
	draw.text((x, y+1), text, fill=shadowcolor)

	# thicker border
	draw.text((x-1, y-1), text, fill=shadowcolor)
	draw.text((x+1, y-1), text, fill=shadowcolor)
	draw.text((x-1, y+1), text, fill=shadowcolor)
	draw.text((x+1, y+1), text, fill=shadowcolor)

	draw.text((x, y), text, fill=fillcolor)

	img.save(file_name)