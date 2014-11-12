#coding:utf-8


##############################################################词典文件读取
#中文路径的处理
path = "C:\\Users\\Syndrome\\Desktop\\语料数据\\360W_字典\\".decode('utf8').encode('cp936') 

myFile = open(path + "dict_360.txt","r")

word_length = []
word_line = []

line = myFile.readline()
i = 0
while line:
	word_line.append(line)
	line = line.rstrip('\n')		#去掉换行符
	m = line.split('\t')		#以\t为分隔符
	#word_length[i] = len(m[0])/3
	word_length.append(len(m[0])/3)
	i += 1
	line = myFile.readline()
	# if i >= 1000:
	# 	break
myFile.close()

print "finish"
print "max of the length of word is " + str(max(word_length))
print len(word_length)
print len(word_line)

#写文件
##############################################################词典文件增加词语长度后，基于长度排序再保存
newPath = path + "dictionary.txt"
myFile = open(newPath , 'w')

for i in range(50,-1,-1):		#for循环的书写
	for j in range(0,len(word_length)):
		if word_length[j] == i:
			newLine = str(i) + '\t' + word_line[j]
			myFile.writelines(newLine)

myFile.close()


##############################################################简化词典文件，基于长度排序的保存
newPath = path + "dictionary_simple.txt"

myFile = open(newPath , 'w')

for i in range(50,-1,-1):		#for循环的书写
	for j in range(0,len(word_length)):
		if word_length[j] == i:
			m = word_line[j].split('\t')
			newLine = m[0] + "\n"
			myFile.writelines(newLine)

myFile.close()

##############################################################词典词语长度坐标文件
new_word_length = sorted(word_length)
new_len = [0]

j = 0
for i in range(0,50):
	while j < len(new_word_length):
		if new_word_length[j] == i:
			pass
		else:
			new_len.append(3669216-j)
			break
		j += 1
new_len.append(3669216-j)

newPath = path + "word_num.txt"

myFile = open(newPath , 'w')

print len(new_len)
print new_len
for i in range(0,len(new_len)):
	myFile.writelines(str(new_len[i]) + '\n')

myFile.close()



