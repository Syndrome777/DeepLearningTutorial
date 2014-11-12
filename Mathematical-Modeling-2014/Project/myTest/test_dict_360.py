#coding:utf-8


path = "C:\\Users\\Syndrome\\Desktop\\语料数据\\360W_字典\\dict_360.txt".decode('utf8').encode('cp936') 

f = open(path,"r")

line = f.readline()
i = 0
while line:
	line = line.rstrip('\n')           #去除字符\n
	m = line.split('\t')            #字符串分割，以\t

	print len(m[0])/3

	for item in m:
		print item             # 后面跟 ',' 将忽略换行符
		# print(line, end = '')　　　# 在 Python 3中使用

	line = f.readline()
	i += 1
	if i == 1000:
		break

f.close()



# 注释代码快捷键，ctrl+/
# def str_len(str):
# 	try:
# 		row_l=len(str)
# 		utf8_l=len(str.encode('utf-8'))
# 		return (utf8_l-row_l)/2+row_l
# 	except:
# 		return None
# 	return None

