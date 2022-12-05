# Import Image from wand.image module
from wand.image import Image
from wand.display import display
import PIL.Image as pilGG
import numpy as np
import io

pil = pilGG.open("doge.jpg")

pil.save("tmp.png")

# Read image using Image() function
with Image(filename="tmp.png") as img:

	# Generate noise image using spread() function
	img.noise("gaussian", attenuate = 1.0)
	# img.save(filename ="gaussiandoge.jpg")
	# display(img)
	img_buffer = np.asarray(bytearray(img.make_blob(format='png')), dtype='uint8')
	bytesio = io.BytesIO(img_buffer)
	pil_img = pilGG.open(bytesio)

	pil_img.show()
