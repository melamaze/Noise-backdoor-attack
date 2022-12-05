
import numpy as np
import os
import cv2
from PIL import Image
from matplotlib import cm


def noisy(noise_typ, image):
    
    image = image.convert('RGB')
    image = np.array(image)
    image = image[:, :, ::-1].copy()

    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        img = Image.fromarray(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
        return img
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        img = Image.fromarray(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
        return img

img = Image.open("doge.jpg")

img = noisy("gauss", img)

img = Image.fromarray(np.uint8(cm.gist_earth(img.transpose(1, 2, 0))*255)).convert('RGB')

img.show()