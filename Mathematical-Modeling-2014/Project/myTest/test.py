#coding:utf-8

import os

path = "C:\\Users\\Syndrome\\Desktop\\语料数据\\文本分类\\20_newsgroups\\".decode("utf-8").encode("cp936")

filenamelist=os.listdir(path)
for item in filenamelist :
	print item
	filenamelist2 = os.listdir(path + "\\" + item)
	for item2 in filenamelist2 :
		print item2
		newPath = path + "\\" + item +"\\" + item2
		myFile = open (newPath)

		myFile.close()

print "finish!"




# myFile = open(path)

# line = myFile.readline()

# while line :
# 	print line
# 	line = myFile.readline()

# myFile.close()




