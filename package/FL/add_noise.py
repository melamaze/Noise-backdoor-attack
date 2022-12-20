import cv2
import numpy as np
from PIL import Image


def add_gaussian(image):

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Split the image into separate color channels
    red_channel, green_channel, blue_channel = cv2.split(image_array)

    # Generate noise matrices for each color channel with standard deviation 4.0
    red_noise = np.random.normal(0, 4.0, red_channel.shape)
    green_noise = np.random.normal(0, 4.0, green_channel.shape)
    blue_noise = np.random.normal(0, 4.0, blue_channel.shape)

    # Add the noise to each color channel
    noisy_red = red_channel + red_noise
    noisy_green = green_channel + green_noise
    noisy_blue = blue_channel + blue_noise

    # Combine the noisy color channels into a single image
    noisy_image = cv2.merge([noisy_red, noisy_green, noisy_blue])

    # Convert the noisy image back to an image and save it
    noisy_image = np.uint8(noisy_image)

    # Convert the image from the OpenCV format to the PIL format
    pil_image = Image.fromarray(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))

    return pil_image


def add_uniform(image):

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Split the image into separate color channels
    red_channel, green_channel, blue_channel = cv2.split(image_array)

    # Generate noise matrices for each color channel with uniform distribution 
    red_noise = np.random.uniform(-2.0, 2.0, red_channel.shape)
    green_noise = np.random.uniform(-2.0, 2.0, green_channel.shape)
    blue_noise = np.random.uniform(-2.0, 2.0, blue_channel.shape)

    # Add the noise to each color channel
    noisy_red = red_channel + red_noise
    noisy_green = green_channel + green_noise
    noisy_blue = blue_channel + blue_noise

    # Combine the noisy color channels into a single image
    noisy_image = cv2.merge([noisy_red, noisy_green, noisy_blue])

    # Convert the noisy image back to an image and save it
    noisy_image = np.uint8(noisy_image)

    # Convert the image from the OpenCV format to the PIL format
    pil_image = Image.fromarray(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))

    return pil_image


def add_salt(image):

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        # Generate a noise matrix with the same shape as the image
    noise = np.zeros(image.shape, dtype=np.uint8)

    # Set a random number of pixels to white (255)
    # num_noise_pixels = np.random.randint(0, high=image.size // 2)
    num_noise_pixels = image.size // 255
    noise[np.random.randint(0, image.shape[0], num_noise_pixels), 
        np.random.randint(0, image.shape[1], num_noise_pixels)] = 255

    # Add the noise to the image
    noisy_image = cv2.add(image, noise)

    # Convert the image from the OpenCV format to the PIL format
    pil_image = Image.fromarray(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))

    return pil_image

def add_gaussian_v2(image):

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    mean = 0
    sigma = 0.06

    # int -> float (標準化)
    image = image / 255
    # 隨機生成高斯 noise (float + float)
    noise = np.random.normal(mean, sigma, image.shape)
    # noise + 原圖
    gaussian_out = image + noise
    # 所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
    gaussian_out = np.clip(gaussian_out, 0, 1)

    # 原圖: float -> int (0~1 -> 0~255)
    gaussian_out = np.uint8(gaussian_out*255)

    noisy_image = gaussian_out

    # noise: float -> int (0~1 -> 0~255)
    noise = np.uint8(noise*255)

    # Convert the image from the OpenCV format to the PIL format
    pil_image = Image.fromarray(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))

    return pil_image