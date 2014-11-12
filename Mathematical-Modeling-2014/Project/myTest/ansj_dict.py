#coding:utf-8

path = "C:\\Users\\Syndrome\\Desktop\\语料数据\\ansj词典\\".decode('utf8').encode('cp936') 
new_path = path + "81W_dict.txt"

#################################################词典读取
myFile = open(new_path,"r")

word_81 = []
word_length = []

line = myFile.readline()

i = 1
while line:
	line = line.rstrip('\n')
	# print line
	word_81.append(line)
	word_length.append(len(line)/3)
	line = myFile.readline()
	i += 1

max_len = max(word_length)
print "the num of word is " + str(i)
print "the max of length is " + str(max_len)
print "part1"

myFile.close()

#################################################词典按长度储存

newPath = path + "ansj_simple.txt"

myFile = open(newPath , 'w')

for i in range(50,-1,-1):		#for循环的书写
	for j in range(0,len(word_length)):
		if word_length[j] == i:
			newLine = word_81[j] + "\n"
			myFile.writelines(newLine)

myFile.close()


print "part2"

##############################################################词典词语长度坐标文件
new_word_length = sorted(word_length)
new_len = [811639]

j = 0
for i in range(0,12):
	while j < len(new_word_length):
		if new_word_length[j] == i:
			pass
		else:
			new_len.append(811639-j)
			break
		j += 1
new_len.append(811639-j)

newPath = path + "ansj_word_num.txt"

myFile = open(newPath , 'w')

print len(new_len)
print new_len
for i in range(0,len(new_len)):
	myFile.writelines(str(new_len[i]) + '\n')

myFile.close()

print "part3"

#################################################分词

word = []

myFile = open(path + "ansj_simple.txt" , 'r')
line = myFile.readline().rstrip('\n')
i = 0
while line:
	word.append(line)
	line = myFile.readline().rstrip('\n')
myFile.close()
print "dictionary is ready!"

word_num = new_len
print "the position of word is ready!"


TEST = "一位朴实美丽的渔家姑娘从红树林边的渔村闯入都市，经历了情感的波折和撞击演绎出复杂而\
又多变的人生。故事发生在有着大面积红树林的小渔村和南海海滨一座新兴的小城里。渔家姑娘珍珠进\
城打工，珍珠公司总经理大虎对她一见钟情，珍珠却不为所动。大虎企图强占珍珠，珍珠毅然回到红树\
林。大虎在另两个干部子弟二虎和三虎的挑唆下，轮奸了珍珠。珍珠的意中人大同进行报复，欲杀大虎\
的母亲、副市长林岚，却刺伤了检查官马叔。大虎又与二虎、三虎轮奸了女工小云，被当场抓获。林岚\
救子心切，落入了刑侦科长金大川手里。马叔与牛晋顶住压力，使案件终于重审，三个虎被绳之以法。"

new_sent = []
T_len = len(TEST)/3

if T_len < 10:
	s = T_len
else:
	s = 9

while s > 0:
	flag = 0
	# print word_num[s]-1
	# print word_num[s+1]
	for i in range(word_num[s]-2,word_num[s+1]-1,-1):
		# print i
		if TEST[0:s*3] == word[i]:
			new_sent.append(word[i])
			print word[i] + "ZZZZZZZZZ"
			flag = 1
			break
	if flag == 1:
		TEST = TEST[s*3:]
		if len(TEST)/3 < 10:
			s = len(TEST)/3
		else:
			s = 9
	else:
		s -= 1
	if s == 1:
		new_sent.append(TEST[:s*3])
		print "TTTTT" + TEST[:s*3] + "    " + str(s)
		TEST = TEST[s*3:]
		if len(TEST)/3 < 10:
			s = len(TEST)/3
		else:
			s = 9
		
for item in new_sent:
	print item + "\\",

print "\npart4"











