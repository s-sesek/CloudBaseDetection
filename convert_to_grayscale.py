from PIL import Image
import os
import math


def dist(red, green, blue):
    return math.sqrt(red**2 + green ** 2 + blue ** 2)


for filename in os.listdir("./convert"):
    image = Image.open("./convert/" + filename)
    pixels = image.load()

    for i in range(image.size[0]):
        for j in range(image.size[1]):

            r, g, b = image.getpixel((i, j))
            distance = dist(r, g, b)

            if distance <= 89:
                pixels[i, j] = (0, 0, 0) #background (black)
            elif distance <= 213:
                pixels[i, j] = (103, 103, 103) #base (gray)
            elif distance <= 324:
                pixels[i, j] = (94, 231, 0)  #sky (green)
            else:
                pixels[i, j] = (231, 231, 231) #side (white)
    image.save("./convert/out.png", "PNG")

