from PIL import Image, ImageFilter
import numpy as np

# 讀入圖片
im = Image.open('doge.jpg')

# 將圖片轉換為 numpy 數組
im_array = np.array(im)

# 生成高斯噪聲
noise = np.random.normal(0, 10, im_array.shape)

# 將噪聲加到圖片中
im_array_noise = (im_array + noise).astype(np.uint8)

# 將 numpy 數組轉換為圖片
im_noise = Image.fromarray(im_array_noise)

# 儲存圖片
im_noise.save('image_noise.jpg')


