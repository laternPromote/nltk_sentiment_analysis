__author__ = 'geyan'
# import ICTCLAS
# traverse the folder with all the comments
# read them into one text file
# split them for further proccessing
# function: read all
import os.path

directory = '\\resources\\original files\\htl_del_4000\\'
negDirectory = '\\resources\\original files\\htl_del_4000\\neg\\'
posDirectory = '\\resources\\original files\\htl_del_4000\\pos\\'


# This method is used to process each file in the directory
def putAllComments(x, dir_name, files):
    print dir_name
    print files
    sentences = []
    for file in files:
        with open(dir_name + file, 'r') as f:
            sentences.append(f.read())
    with open(directory + x + '.txt', 'w+') as negFile:
        for sentence in sentences:
            negFile.write(sentence)
            # print sentences


if os.path.exists(negDirectory):
    os.path.walk(negDirectory, putAllComments, 'neg')
else:
    print "negDirectory doesn't exist"
if os.path.exists(posDirectory):
    os.path.walk(posDirectory, putAllComments, 'pos')
