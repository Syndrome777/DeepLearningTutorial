#coding:utf-8

path = "C:\\Users\\Syndrome\\Desktop\\语料数据\\360W_字典\\".decode('utf8').encode('cp936')

newPath = path + "dictionary_simple.txt"

word = []
word_num = []

####################################################################最简字典文件打开并进入内存
myFile = open(newPath , 'r')

line = myFile.readline().rstrip('\n')
i = 0
while line:
	word.append(line)
	line = myFile.readline().rstrip('\n')
	# if i == 2000:
	# 	print word[i]
	# i=i+1

myFile.close()
print len(word)
print "part1"

####################################################################词典词语长度坐标文件进去内存
newPath2 = path + "word_num.txt"
myFile = open(newPath2 , 'r')

line = myFile.readline().rstrip('\n')

while line:
	word_num.append(int(line))
	line = myFile.readline().rstrip('\n')

myFile.close()

print len(word_num)

print "part2"
####################################################################利用词典进行分词

TEST = "你是我一生的挚爱啊我的女神"

TEST = "一位朴实美丽的渔家姑娘从红树林边的渔村闯入都市，经历了情感的波折和撞击演绎出复杂而\
又多变的人生。故事发生在有着大面积红树林的小渔村和南海海滨一座新兴的小城里。渔家姑娘珍珠进\
城打工，珍珠公司总经理大虎对她一见钟情，珍珠却不为所动。大虎企图强占珍珠，珍珠毅然回到红树\
林。大虎在另两个干部子弟二虎和三虎的挑唆下，轮奸了珍珠。珍珠的意中人大同进行报复，欲杀大虎\
的母亲、副市长林岚，却刺伤了检查官马叔。大虎又与二虎、三虎轮奸了女工小云，被当场抓获。林岚\
救子心切，落入了刑侦科长金大川手里。马叔与牛晋顶住压力，使案件终于重审，三个虎被绳之以法。"


new_sent = []

T_len = len(TEST)/3

if T_len < 41:
	s = T_len
else:
	s = 40

while s > 0:
	flag = 0
	# print word_num[s]-1
	# print word_num[s+1]
	# print s
	# print TEST[0:s*3]
	for i in range(word_num[s]-1,word_num[s+1],-1):
		#print word[i]
		if TEST[0:s*3] == word[i]:
			new_sent.append(word[i])
			print word[i] + "ZZZZZZZZZ"
			flag = 1
			break
	if flag == 1:
		TEST = TEST[s*3:]
		if len(TEST)/3 < 41:
			s = len(TEST)/3
		else:
			s = 40
	else:
		s -= 1
	if s == 1:
		new_sent.append(TEST[:s*3])
		print "TTTTT" + TEST[:s*3] + "    " + str(s)
		TEST = TEST[s*3:]
		if len(TEST)/3 < 41:
			s = len(TEST)/3
		else:
			s = 40
		

for item in new_sent:
	print item + "\\",


print "\npart3"
















