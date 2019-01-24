from PIL import Image
import os

for filename in os.listdir("./convert"):
    im = Image.open("./convert/" + filename)
    out = im.convert('P', palette=Image.ADAPTIVE, colors=5).convert('LA')
    out.convert('LA')
    out.save("./convert/" + filename, "PNG")
