# Import Image from wand.image module
from wand.image import Image
import PIL.Image as pilGG
import numpy as np
import io


def add_gaussian_noise(im):

    pil_img = im
    im.save("tmp.png")

    # Read image using Image() function
    with Image(filename="tmp.png") as img:
    
        # Generate noise image using spread() function
        img.noise("gaussian", attenuate = 0.9)

        # wand to PIL
        img_buffer = np.asarray(bytearray(img.make_blob(format='png')), dtype='uint8')
	    bytesio = io.BytesIO(img_buffer)
	    pil_img = pilGG.open(bytesio)

    return pil_img