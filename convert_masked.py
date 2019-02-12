import cv2
import math
import numpy as np
import os



def main():
    white = np.array([231, 231, 231])
    gray = np.array([103, 103, 103])
    black = np.array([0, 0, 0])
    green = np.array([1, 231, 95])

    label_background = np.array([0, 0, 0])
    label_base = np.array([1, 1, 1])
    label_side = np.array([2, 2, 2])
    label_sky = np.array([3, 3, 3])

    for filename in os.listdir("./trainingdata/maskedimages"):
        if filename.endswith('.png'):

            print("Converting " + filename)

            img = cv2.imread("./trainingdata/maskedimages/" + filename)
            height, width, channels = img.shape
            for x in range(0, height - 1):
                for y in range(0, width - 1):

                    distance_white = calculate_distance_to_color(img[x][y], white)
                    distance_black = calculate_distance_to_color(img[x][y], black)
                    distance_green = calculate_distance_to_color(img[x][y], green)
                    distance_gray = calculate_distance_to_color(img[x][y], gray)

                    closest_color = min([distance_white, distance_black, distance_green, distance_gray])

                    if closest_color == distance_black:
                        convert_color(img[x][y], label_background)
                    elif closest_color == distance_green:
                        convert_color(img[x][y], label_base)
                    elif closest_color == distance_white:
                        convert_color(img[x][y], label_side)
                    else:
                        convert_color(img[x][y], label_sky)

            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            replacemaskunderscore = str.replace(filename, "mask", "_")
            cv2.imwrite("./trainingdata/groundtruthimages/" + replacemaskunderscore, gray_image)


def calculate_distance_to_color(pixel, color):
    distance = math.sqrt((pixel[0]-color[0])**2+(pixel[1]-color[1])**2+(pixel[2]-color[2])**2)
    return distance


def convert_color(pixel, color):
    pixel[0] = color[0]
    pixel[1] = color[1]
    pixel[2] = color[2]


main()
