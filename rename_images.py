import os


for filename in os.listdir("./trainingdata/images"):
    if filename.endswith('.png'):
        replacemaskunderscore = str.replace(filename, "mask", "_")
        os.rename("./trainingdata/images/" + filename, "./trainingdata/images/" + replacemaskunderscore)
