import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm

# img = cv2.imread("doge.jpg")[...,::-1]/255.0

pil = Image.open('doge.jpg').convert('RGB')
open_cv_image = np.array(pil) 
# Convert RGB to BGR 
img = (open_cv_image[:, :, ::-1]/255.0).copy() 

noise =  np.random.normal(loc=0, scale=1, size=img.shape)

# noise overlaid over image
# noisyy = np.clip((img + noise*0.4), 0, 1)
noisy = np.clip((img + noise*0.2),0,1)
noisy2 = np.clip((img + noise*0.4),0,1)

# noise multiplied by image:
# whites can go to black but blacks cannot go to white
noisy2mul = np.clip((img*(1 + noise*0.2)),0,1)
noisy4mul = np.clip((img*(1 + noise*0.4)),0,1)

noisy2mul = np.clip((img*(1 + noise*0.2)),0,1)
noisy4mul = np.clip((img*(1 + noise*0.4)),0,1)

# noise multiplied by bottom and top half images,
# whites stay white blacks black, noise is added to center
img2 = img*2
n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.2)), (1-img2+1)*(1 + noise*0.2)*-1 + 2)/2, 0,1)
n4 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.4)), (1-img2+1)*(1 + noise*0.4)*-1 + 2)/2, 0,1)


# cv2.imshow('qq', noisy2)
# cv2.waitKey(0)

ret = Image.fromarray(cv2.cvtColor(noisy2.astype('uint8') * 255, cv2.COLOR_BGR2RGB))

ret.show()

# norm noise for viz only
# noise2 = (noise - noise.min())/(noise.max()-noise.min())
# plt.figure(figsize=(20,20))
# plt.imshow(np.vstack((np.hstack((img, noise2)),
#                       np.hstack((noisy, noisy2)),
#                       np.hstack((noisy2mul, noisy4mul)),
#                       np.hstack((n2, n4)))))
# plt.show()
# plt.hist(noise.ravel(), bins=100)
# plt.show()