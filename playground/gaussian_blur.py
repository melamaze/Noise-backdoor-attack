# import cv2
# import numpy as np

# # read the image
# image = cv2.imread('noisy_image.jpg')

# # apply a Gaussian blur to the image
# blur = cv2.GaussianBlur(image, (5, 5), 0)

# # save the denoised image
# cv2.imwrite('denoised_image.jpg', blur)


import cv2
import numpy as np
import glob as glob

cnt = 0

for img_path in glob.glob('/home/hentci/code/noise_cifar/cat_trig/*'):


    # name = img_path[-8:]
    # print(name)
    name = str(cnt) + '.jpg'

    img = cv2.imread(img_path)

    # denoise = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.imwrite('/home/hentci/code/denoise_cifar/cat_trig/' + name, blur)

    cnt += 1