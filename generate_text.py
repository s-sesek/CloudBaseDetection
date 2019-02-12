import os

# start editable vars #
train	= "./trainingdata/train.txt"	# file to save the results to
test	= "./trainingdata/test.txt"	# file to save the results to
eval	= "./trainingdata/val.txt"	# file to save the results to


with open(train, "w") as trainfile:
    with open(test, "w") as testfile:
        with open (eval, "w") as valfile:

            counter = 0
            for file in os.listdir("./trainingdata/images"):
                filename = file.split(".")[0]

                # Write all files to eval
                valfile.write("%s\n" % filename)

                if counter > 7:
                    testfile.write("%s\n" % filename)
                    if counter == 9:
                        counter = 0
                else:
                    trainfile.write("%s\n" % filename)

                counter += 1

trainfile.close()
testfile.close()
valfile.close()
