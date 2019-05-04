import cv2
import numpy as np

BACK_GROUND = 0
BASE = 3
SIDE = 2
SKY = 1

filename = '14400_005.png'

groundTruth = cv2.imread("./eval/truth/" + filename)
output = cv2.imread("./eval/prediction/" + filename)

# - Pixel labels
# label_background = np.array([0, 0, 0])
# label_base = np.array([1, 1, 1])
# label_side = np.array([2, 2, 2])
# label_sky = np.array([3, 3, 3])
# [95 231  1] - Sky
# [0 0 0] - Background
# [103 103 103] - Bottom
# [231 231 231]- Side


# - Convert a prediction color back to it's ground truth label
# - Could probably use single digits instead of array?
def convert_prediction_to_ground_truth(color):
    if color[0] == 95:
        return np.array([1, 1, 1])  # - Sky
    if color[0] == 0:
        return np.array([0, 0, 0])  # - Background
    if color[0] == 103:
        return np.array([3, 3, 3])  # - Base
    if color[0] == 231:
        return np.array([2, 2, 2])  # - Side
    else:
        raise Exception("Error in colors")


# Output and ground truth should have the same shape
height, width, channels = groundTruth.shape

mismatch = 0
total = 0

baseAsSide = 0
baseAsBackground = 0
baseAsSky = 0
baseAsBase = 0

skyAsSide = 0
skyAsBackground = 0
skyAsBase = 0
skyAsSky = 0

sideAsSide = 0
sideAsBackground = 0
sideAsBase = 0
sideAsSky = 0

bgAsSide = 0
bgAsBg = 0
bgAsBase = 0
bgAsSky = 0

# Iterate over each x & y pixel
for x in range(0, height - 1):
    for y in range(0, width - 1):

        maskedColor = groundTruth[x][y]
        color = output[x][y]
        print(maskedColor,  "  ", color)

        predictionColor = convert_prediction_to_ground_truth(output[x][y])

        # Sky
        if maskedColor[0] == SKY and predictionColor[0] == BACK_GROUND:
            skyAsBackground += 1
        if maskedColor[0] == SKY and predictionColor[0] == BASE:
            skyAsBase += 1
        if maskedColor[0] == SKY and predictionColor[0] == SIDE:
            skyAsSide += 1
        if maskedColor[0] == SKY and predictionColor[0] == SKY:
            skyAsSky += 1

        # Side
        if maskedColor[0] == SIDE and predictionColor[0] == BACK_GROUND:
            sideAsBackground += 1
        if maskedColor[0] == SIDE and predictionColor[0] == BASE:
            sideAsBase += 1
        if maskedColor[0] == SIDE and predictionColor[0] == SIDE:
            sideAsSide += 1
        if maskedColor[0] == SIDE and predictionColor[0] == SKY:
            sideAsSky += 1
        
        # Base
        if maskedColor[0] == BASE and predictionColor[0] == BACK_GROUND:
            baseAsBackground += 1
        if maskedColor[0] == BASE and predictionColor[0] == BASE:
            baseAsBase += 1
        if maskedColor[0] == BASE and predictionColor[0] == SIDE:
            baseAsSide += 1
        if maskedColor[0] == BASE and predictionColor[0] == SKY:
            baseAsSky += 1

        # Background
        if maskedColor[0] == BACK_GROUND and predictionColor[0] == BACK_GROUND:
            bgAsBg += 1
        if maskedColor[0] == BACK_GROUND and predictionColor[0] == BASE:
            bgAsBase += 1
        if maskedColor[0] == BACK_GROUND and predictionColor[0] == SIDE:
            bgAsSide += 1
        if maskedColor[0] == BACK_GROUND and predictionColor[0] == SKY:
            bgAsSky += 1


print("---------------")
print("BASE as background:" + str(baseAsBackground))
print("BASE as side:" + str(baseAsSide))
print("BASE as sky:" + str(baseAsSky))
print("BASE as base:" + str(baseAsBase))
baseTotal = baseAsBackground + baseAsSide + baseAsSky + baseAsBase
baseAccuracy = (baseAsBase/baseTotal) * 100
print("BASE ACCURACY:" + str(baseAccuracy))


print("---------------")
print("SKY as background:" + str(skyAsBackground))
print("SKY as side:" + str(skyAsSide))
print("SKY as sky:" + str(skyAsSky))
print("SKY as base:" + str(skyAsBase))
skyTotal = skyAsBackground + skyAsSide + skyAsSky + skyAsBase
skyAccuracy = (skyAsSky/skyTotal) * 100
print("SKY ACCURACY:" + str(skyAccuracy))

print("---------------")
print("SIDE as background:" + str(sideAsBackground))
print("SIDE as side:" + str(sideAsSide))
print("SIDE as sky:" + str(sideAsSky))
print("SIDE as base:" + str(sideAsBase))
sideTotal = sideAsBackground + sideAsSide + sideAsSky + sideAsBase
sideAccuracy = (sideAsSide/sideTotal) * 100
print("SIDE ACCURACY:" + str(sideAccuracy))

print("---------------")
print("BACKGROUND as background:" + str(bgAsBg))
print("BACKGROUND as side:" + str(bgAsSide))
print("BACKGROUND as sky:" + str(bgAsSky))
print("BACKGROUND as base:" + str(bgAsBase))
bgTotal = bgAsBg + bgAsSide + bgAsSky + bgAsBase
bgAccuracy = (bgAsBg/bgTotal) * 100
print("BG ACCURACY:" + str(bgAccuracy))
