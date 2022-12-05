# Import Image from wand.image module
from wand.image import Image

# Read image using Image() function
with Image(filename ="doge.jpg") as img:

	# Generate noise image using spread() function
	img.noise("gaussian", attenuate = 1.0)
	img.save(filename ="gaussiandoge.jpg")
