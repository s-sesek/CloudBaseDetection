import os

# start editable vars #
outputfile	= "inventory.txt"	# file to save the results to
exclude		= ['Thumbs.db','.tmp']	# exclude files containing these strings
pathsep		= "/"			# path seperator ('/' for linux, '\' for Windows)
# end editable vars #

with open(outputfile, "w") as txtfile:
    for file in os.listdir("./trainingdata/maskedimages"):
        filename = file.split(".")[0]
        txtfile.write("%s\n" % filename)

txtfile.close()
